# Despejo de KV-cache em LLMs com Aprendizagem por Reforço

Projeto da disciplina **GBC063 — Inteligência Artificial** (FACOM/UFU), Prof. Albertini.

**Dupla:** Ihury Kewin · Willian Casa Grandi

Este repositório reúne as entregas do projeto ao longo do semestre. A primeira entrega (Desafio 1) é a contextualização do problema; o desenvolvimento acompanha o conteúdo da disciplina.

---

## O problema

Quando um LLM gera texto, ele guarda as chaves e valores (K, V) de todos os tokens anteriores para não recalcular a atenção. Esse **KV-cache** cresce linearmente com o tamanho do contexto e, em textos longos, chega a ocupar mais memória de GPU que os próprios pesos do modelo.

A saída usual é fixar um orçamento de memória (por exemplo, 16 blocos de 16 tokens) e descartar o resto. A pergunta é **quais blocos descartar**. Hoje isso é feito com regras fixas:

| Heurística | Regra |
|---|---|
| Janela deslizante | mantém só os mais recentes |
| StreamingLLM | mantém os primeiros tokens ("attention sinks") + os recentes |
| H2O (Heavy-Hitter Oracle) | mantém os que mais receberam atenção até agora |

A proposta é tratar essa escolha como um **processo de decisão sequencial** e aprender a política com aprendizagem por reforço.

## Modelagem (Desafio 1)

**Episódio:** o modelo lê um texto longo bloco a bloco, com no máximo `B` blocos no cache. Os primeiros `B` entram livremente; a partir daí, a cada bloco novo, o agente escolhe um dos `B + 1` candidatos para despejar.

### 1. Ambiente e estado

Para cada bloco candidato, o agente observa:

| Feature | Descrição |
|---|---|
| idade | há quantos blocos ele entrou no cache |
| atenção acumulada | atenção média recebida desde que entrou |
| atenção recente | atenção média nos últimos `k` tokens |
| sink | se é um dos blocos iniciais do texto |
| posição | posição relativa no texto |

Mais o progresso global no texto (`t / T`).

**Sobre a propriedade de Markov:** só a atenção do último passo não é Markov, porque um bloco pode ficar "adormecido" e voltar a ser importante. A atenção acumulada e a janela recente resumem a história e aproximam uma estatística suficiente. Mesmo assim, a importância futura de um bloco depende de texto que ainda não chegou, então formalmente o problema é um **POMDP**.

### 2. Ações

`a_t ∈ {0, 1, …, B}`: o índice do candidato a despejar (os `B` blocos do cache ou o próprio bloco novo). Um bloco despejado nunca volta.

Extensão prevista: uma ação de "comprimir" (quantizar para 4 bits) em vez de descartar.

### 3. Recompensas

- **Densa:** `r_t = −KL(p_cheio ‖ p_reduzido)`, a divergência entre a distribuição do próximo token com cache completo e com o cache reduzido.
- **Densa, versão barata para treino:** `r_t = −` massa de atenção que os blocos já despejados receberiam nos próximos tokens, calculada a partir de atenções gravadas.
- **Terminal:** `+10` se a "agulha" (uma senha escondida no início do texto e perguntada no final) ainda estiver no cache.
- A restrição de memória é dura (o cache nunca passa de `B`), então não há penalidade de estouro.

## Desafio 2: DQN no ambiente proxy

### Como o ambiente funciona

Rodar o LLM a cada decisão de treino seria lento demais. Por isso usamos um **ambiente proxy**:

1. O GPT-2 lê cada texto **uma única vez**, com o cache completo, e a atenção de cada token para cada bloco (média sobre camadas e cabeças) é gravada num *trace*.
2. O ambiente (`KVCacheEnv`, Gymnasium) simula os despejos a partir do trace: a cada bloco novo, o agente escolhe quem sai e perde a atenção que os próximos tokens dariam aos blocos já apagados.
3. Cada texto tem uma **agulha** ("the secret code is NNNN") inserida em posição aleatória no primeiro quarto do texto, fora do bloco inicial, e uma pergunta sobre ela no final.

Os textos vêm da WikiText-2, com 1.024 tokens cada (64 blocos de 16, orçamento de 16 blocos, 47 decisões por texto): 200 textos de treino e 50 de teste, sem sobreposição. Há também uma fonte **sintética** que imita esses padrões e roda sem GPU nem internet.

### DQN

Stable-Baselines3 com `MlpPolicy` (rede 128×128): 86 entradas, 17 saídas, Q(s, a) de despejar cada candidato. Hiperparâmetros em [`configs/dqn.yml`](configs/dqn.yml), no mesmo formato do `dqn.yml` da atividade de Atari:

| Parâmetro | Valor | Por quê |
|---|---|---|
| `n_timesteps` | 200.000 | ~4.250 textos de 47 decisões |
| `learning_starts` | 5.000 | episódios curtos; não precisa de 100 mil passos aleatórios |
| `exploration_fraction` | 0,3 (ε de 1,0 a 0,05) | explorar mais para descobrir o prêmio da agulha |
| `buffer_size`, `learning_rate`, `target_update_interval`, `train_freq` | 100k, 1e-4, 1.000, 4 | iguais ao Atari |

A cada 5.000 passos, a política gulosa (sem exploração) é avaliada nos 50 textos de teste. Treinamos com 3 sementes.

### Como rodar

Requer Python 3.10+ (no Debian/Ubuntu: `sudo apt install python3-venv`).

```bash
make setup-gpt2      # cria .venv e instala tudo (PyTorch, SB3, transformers)
make teste           # testes do ambiente
make tudo            # traces do GPT-2 -> baselines -> DQN (3 sementes) -> gráficos
```

Sem GPU ou sem internet: `make setup && make tudo FONTE=sintetico`.

Etapa por etapa (o que o `make tudo` faz):

```bash
.venv/bin/python scripts/gerar_traces_gpt2.py          # dados/gpt2/{treino,teste}
.venv/bin/python scripts/avaliar_baselines.py --fonte gpt2
.venv/bin/python scripts/treinar_dqn.py --fonte gpt2    # resultados/gpt2/dqn/seed{0,1,2}
.venv/bin/python scripts/graficos.py --fonte gpt2       # resultados/gpt2/figuras
```

### Saídas

| Arquivo | Conteúdo |
|---|---|
| `resultados/<fonte>/baselines.csv` | retorno, atenção perdida e agulha de cada política de referência |
| `resultados/<fonte>/dqn/seed*/curva.csv` | avaliação no teste a cada 5.000 passos |
| `resultados/<fonte>/dqn/seed*/final.json` | avaliação do modelo final |
| `resultados/<fonte>/figuras/curva_aprendizado.png` | retorno no teste ao longo do treino, com as políticas de referência |
| `resultados/<fonte>/figuras/comparacao_final.png` | atenção perdida e agulha mantida por política |

### Estrutura

```
configs/dqn.yml                  hiperparâmetros do DQN
src/kvcache_rl/
  env.py                         ambiente Gymnasium (KVCacheEnv)
  traces.py                      formato dos traces e gerador sintético
  gpt2.py                        gravação da atenção do GPT-2 e inserção da agulha
  baselines.py                   janela deslizante, StreamingLLM, H2O, aleatória, oráculo
scripts/                         gerar traces, avaliar, treinar, gráficos
tests/                           testes do ambiente
```

## Entregas

- [x] **Desafio 1** — apresentação do problema: ambiente/estado, ações e recompensas — 29/10/2026 ([slides](apresentacoes/desafio1/))
- [ ] **Desafio 2** — DQN no problema e gráficos dos resultados — 12/11/2026

## Referências

- Xiao et al. *Efficient Streaming Language Models with Attention Sinks* (StreamingLLM). ICLR 2024. [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
- Zhang et al. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models*. NeurIPS 2023. [arXiv:2306.14048](https://arxiv.org/abs/2306.14048)
- Li et al. *SnapKV: LLM Knows What You are Looking for Before Generation*. 2024. [arXiv:2404.14469](https://arxiv.org/abs/2404.14469)
- Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM). SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- Mnih et al. *Human-level control through deep reinforcement learning* (DQN). Nature, 2015.
- Merity et al. *Pointer Sentinel Mixture Models* (WikiText). 2016. [arXiv:1609.07843](https://arxiv.org/abs/1609.07843)
- Sutton & Barto. *Reinforcement Learning: An Introduction*, 2ª ed., 2018.

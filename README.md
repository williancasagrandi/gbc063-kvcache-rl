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

1. O GPT-2 lê cada texto **uma única vez**, com o cache completo, e a atenção é gravada num *trace*: para cada cabeça de atenção (12 camadas × 12 cabeças), quanto os tokens de cada bloco consultam cada bloco anterior. O trace guarda também as médias por camada e do modelo todo.
2. O ambiente (`KVCacheEnv`, Gymnasium) simula os despejos a partir do trace: a cada bloco novo, o agente escolhe quem sai e perde a atenção que os próximos tokens dariam aos blocos já apagados.
3. Cada texto tem uma **agulha** ("the secret code is NNNN") inserida em posição aleatória no primeiro quarto do texto, fora do bloco inicial, e uma pergunta sobre ela no final.

Como o agente só decide nas fronteiras de bloco, o trace guarda a atenção **média dos tokens de cada bloco**, e não token a token: a perda e todas as features saem exatamente iguais (há um teste que confere isso contra uma simulação token a token) e o arquivo fica 16 vezes menor.

Os textos vêm da WikiText-2, com 1.024 tokens cada (64 blocos de 16, orçamento de 16 blocos por cache): 200 textos de treino e 50 de teste, sem sobreposição. Há também uma fonte **sintética** que imita esses padrões, com 144 cabeças de papéis diferentes (locais, de sink, de blocos importantes, de revisita e poucas "de recuperação" que buscam a agulha), e roda sem GPU nem internet.

### Três modos: geral, por camada e por cabeça

Cada camada e cada cabeça do GPT-2 presta atenção em coisas diferentes, e no GPT-2 cada cabeça tem o seu próprio K e V, ou seja, o seu próprio pedaço do cache. Testamos três granularidades, sempre com **o mesmo orçamento (16 blocos) em cada cache**:

| Modo | Caches | O agente vê, por candidato | Episódio | Entradas da rede |
|---|---|---|---|---|
| `geral` | 1, igual para o modelo todo | idade, sink, posição, atenção acumulada e recente **na média do modelo** | um texto | 86 |
| `por_camada` | 12, um por camada | o mesmo, com a atenção **daquela camada** + qual camada é | um texto em uma camada | 98 |
| `por_cabeca` | 144, um por cabeça | o mesmo, com a atenção **daquela cabeça** + qual cabeça é | um texto em uma cabeça | 230 |

Nos modos por camada e por cabeça, **o mesmo agente serve para todas as unidades** (parâmetros compartilhados) e cada unidade é um episódio separado. Os caches das unidades não interagem: se elas decidissem em sequência no mesmo episódio, o "próximo estado" de uma decisão seria o cache de outra unidade, que o agente não vê, e a propriedade de Markov se perderia. Para avaliar um texto, rodamos todas as unidades e tiramos a média. As regras fixas também passam a decidir por unidade (o H2O por cabeça é o método original).

**A agulha só importa para quem a consulta.** Poucas cabeças buscam informação distante no contexto (as *retrieval heads*). Por isso, em cada unidade, o bônus da agulha é `10 ×` a relevância dela: a atenção que a pergunta final dá à agulha naquela unidade, dividida pela da unidade que mais a consulta. A métrica "agulha mantida" de um texto é a fração dessa atenção que foi preservada: no modo geral é 0 ou 1; por camada e por cabeça, a média ponderada das unidades. Assim, **atenção perdida e agulha mantida são comparáveis entre os modos**. O retorno não é, porque a escala do bônus muda, então ele só é comparado dentro de cada modo.

### DQN

Stable-Baselines3 com `MlpPolicy`: uma saída Q(s, a) para cada um dos 17 candidatos. Hiperparâmetros em [`configs/dqn.yml`](configs/dqn.yml), no mesmo formato do `dqn.yml` da atividade de Atari, com uma entrada por modo:

| Parâmetro | `geral` | `por_camada` | `por_cabeca` | Por quê |
|---|---|---|---|---|
| `n_timesteps` | 200.000 | 300.000 | 500.000 | ~4.250, ~6.400 e ~10.600 episódios de 47 decisões |
| `policy_kwargs.net_arch` | 128×128 | 128×128 | 256×256 | por cabeça o agente aprende o papel de 144 unidades |
| `learning_starts` | 5.000 | 5.000 | 5.000 | episódios curtos; não precisa de 100 mil passos aleatórios |
| `exploration_fraction` | 0,3 | 0,3 | 0,3 | ε de 1,0 a 0,05; explorar mais para achar o prêmio da agulha |
| `buffer_size` | 100k | 100k | 200k | |
| `learning_rate`, `train_freq`, `target_update_interval` | 1e-4, 4, 1.000 | iguais | iguais | como no Atari |

A política gulosa (sem exploração) é avaliada nos 50 textos de teste, em todas as unidades, a cada 5.000 / 10.000 / 25.000 passos. Treinamos com 3 sementes.

### Como rodar

Requer Python 3.10+ (no Debian/Ubuntu: `sudo apt install python3-venv`).

```bash
make setup-gpt2      # cria .venv e instala tudo (PyTorch, SB3, transformers)
make teste           # testes do ambiente
make tudo-modos      # traces do GPT-2 -> os três modos (baselines, DQN, gráficos) -> comparação
```

Um modo só: `make tudo MODO=por_cabeca` (ou `make baselines treinar graficos MODO=por_camada` sem regerar os traces). Sem GPU ou sem internet: `make setup && make tudo-modos FONTE=sintetico`.

Etapa por etapa:

```bash
.venv/bin/python scripts/gerar_traces_gpt2.py                              # dados/gpt2/{treino,teste}
.venv/bin/python scripts/avaliar_baselines.py --fonte gpt2 --modo por_cabeca
.venv/bin/python scripts/treinar_dqn.py --fonte gpt2 --modo por_cabeca      # resultados/gpt2/por_cabeca/dqn/seed{0,1,2}
.venv/bin/python scripts/graficos.py --fonte gpt2 --modo por_cabeca         # resultados/gpt2/por_cabeca/figuras
.venv/bin/python scripts/comparar_modos.py --fonte gpt2                     # depois de rodar os três modos
```

Tempo aproximado num notebook (CPU): traces do GPT-2 de 2 a 10 min; DQN por semente cerca de 2,5 min no `geral`, 4 min no `por_camada` e 12 min no `por_cabeca`. O `make tudo-modos` inteiro leva cerca de 1 hora.

### Saídas

| Arquivo | Conteúdo |
|---|---|
| `resultados/<fonte>/<modo>/baselines.csv` | retorno, atenção perdida e agulha de cada política de referência |
| `resultados/<fonte>/<modo>/dqn/seed*/curva.csv` | avaliação no teste ao longo do treino |
| `resultados/<fonte>/<modo>/dqn/seed*/final.json` | avaliação do modelo final (= último ponto da curva) |
| `resultados/<fonte>/<modo>/figuras/curva_aprendizado.png` | retorno no teste ao longo do treino, com as políticas de referência |
| `resultados/<fonte>/<modo>/figuras/comparacao_final.png` | atenção perdida e agulha mantida por política |
| `resultados/<fonte>/comparacao_modos.png` e `.csv` | H2O, DQN e oráculo nos três modos |

### Estrutura

```
configs/dqn.yml                  hiperparâmetros do DQN, um bloco por modo
src/kvcache_rl/
  env.py                         ambiente Gymnasium (KVCacheEnv) com os três modos
  traces.py                      formato dos traces (por bloco; modelo, camada e cabeça) e gerador sintético
  gpt2.py                        gravação da atenção do GPT-2 por cabeça e inserção da agulha
  baselines.py                   janela deslizante, StreamingLLM, H2O, aleatória, oráculo e avaliação
  config.py, visual.py           leitura do dqn.yml e estilo dos gráficos
scripts/                         gerar traces, avaliar, treinar, gráficos, comparar modos
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
- Wu et al. *Retrieval Head Mechanistically Explains Long-Context Factuality*. 2024. [arXiv:2404.15574](https://arxiv.org/abs/2404.15574)
- Xiao et al. *DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads*. 2024. [arXiv:2410.10819](https://arxiv.org/abs/2410.10819)
- Fu et al. *Not All Heads Matter: A Head-Level KV Cache Compression Method with Integrated Retrieval and Reasoning* (HeadKV). 2024. [arXiv:2410.19258](https://arxiv.org/abs/2410.19258)
- Mnih et al. *Human-level control through deep reinforcement learning* (DQN). Nature, 2015.
- Merity et al. *Pointer Sentinel Mixture Models* (WikiText). 2016. [arXiv:1609.07843](https://arxiv.org/abs/1609.07843)
- Sutton & Barto. *Reinforcement Learning: An Introduction*, 2ª ed., 2018.

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

1. O LLM lê cada texto **uma única vez**, com o cache completo, e a atenção é gravada num *trace*: para cada cabeça de K/V, quanto os tokens de cada bloco consultam cada bloco anterior. O trace guarda também as médias por camada e do modelo todo.
2. O ambiente (`KVCacheEnv`, Gymnasium) simula os despejos a partir do trace: a cada bloco novo, o agente escolhe quem sai e perde a atenção que os próximos tokens dariam aos blocos já apagados. **A recompensa é só isso: menos a atenção perdida.** O retorno de um episódio é menos a atenção perdida acumulada. Cada decisão é cobrada pelo próprio custo: a atenção que o bloco despejado receberia dali até o fim do texto (ver [DQN](#dqn)).
3. Cada texto tem uma **agulha** ("the secret code is NNNN") inserida em posição aleatória no primeiro quarto do texto, fora do bloco inicial, e uma pergunta sobre ela no final. Ela **não entra na recompensa**: serve só para testar, no fim, se o modelo ainda recupera uma informação distante.
4. Ao final, cada política é avaliada **no próprio LLM**, com o cache podado: quanto a perplexidade do texto sobe e se o modelo ainda acerta a senha (ver [Métricas](#métricas-o-que-medimos-e-por-quê)).

Como o agente só decide nas fronteiras de bloco, o trace guarda a atenção **média dos tokens de cada bloco**, e não token a token: a perda e todas as features saem exatamente iguais (há um teste que confere isso contra uma simulação token a token) e o arquivo fica 16 vezes menor.

Os textos vêm da WikiText-2, com 1.024 tokens cada (64 blocos de 16, orçamento de 16 blocos por cache): 200 textos de treino e 50 de teste, sem sobreposição. Há também uma fonte **sintética** que imita esses padrões, com cabeças de papéis diferentes (locais, de sink, de blocos importantes, de revisita e poucas "de recuperação" que buscam a agulha), e roda sem GPU nem internet.

Antes de gravar a atenção de um texto, o script pergunta a senha ao modelo **com o cache completo** e anota se ele acerta. Essa taxa diz se a tarefa faz sentido para aquele modelo; com `--exigir-acerto`, só os textos em que ele acerta viram trace.

### Dois modelos: GPT-2 e Qwen2.5-0.5B

O cache guarda um K e um V por **cabeça de K/V**, e é essa a unidade que pode ser despejada:

| Modelo | Camadas | Consultas (Q) | Cabeças de K/V | Caches | Atenção |
|---|---|---|---|---|---|
| GPT-2 small (`--fonte gpt2`) | 12 | 12 | 12 | 12 × 12 = **144** | MHA: cada consulta tem o seu K/V |
| Qwen2.5-0.5B (`--fonte qwen`) | 24 | 14 | 2 | 24 × 2 = **48** | GQA: 7 consultas compartilham cada K/V |

O Qwen2.5-0.5B é um modelo atual e usa **GQA** (*grouped-query attention*), hoje o padrão (Llama 3, Mistral, Qwen): várias cabeças de consulta compartilham o mesmo K/V, justamente para o cache caber na memória. Como um token só sai do cache para o grupo inteiro, a atenção das 7 consultas de um grupo é mediada ao gravar o trace — é o que aquele cache serve, em média. O script lê camadas e cabeças da configuração do modelo e imprime, então trocar de modelo não exige mexer no código.

### Três modos: geral, por camada e por cabeça

Cada camada e cada cabeça presta atenção em coisas diferentes. Os modos não mudam **onde** o cache fica (sempre nas cabeças de K/V), e sim **com que granularidade a política decide**: no modo geral, uma decisão é aplicada a todas as cabeças de K/V; por camada, a decisão de uma camada vale para as cabeças de K/V dela. Os modos mais grossos são os mais fáceis de implementar de verdade — se todas as cabeças apagam os mesmos tokens, o tensor do cache continua retangular, que é como o vLLM gerencia memória — e a escada entre eles mede quanto vale a liberdade extra. Sempre com **o mesmo orçamento (16 blocos) em cada cache**:

| Modo | Caches | O agente vê, por candidato | Episódio | Entradas da rede (GPT-2 / Qwen) |
|---|---|---|---|---|
| `geral` | 1, igual para o modelo todo | idade, sink, posição, atenção acumulada e recente **na média do modelo** | um texto | 86 / 86 |
| `por_camada` | um por camada (12 / 24) | o mesmo, com a atenção **daquela camada** + qual camada é | um texto em uma camada | 98 / 110 |
| `por_cabeca` | um por cabeça de K/V (144 / 48) | o mesmo, com a atenção **daquela cabeça** + qual cabeça é | um texto em uma cabeça | 230 / 134 |

Nos modos por camada e por cabeça, **o mesmo agente serve para todas as unidades** (parâmetros compartilhados) e cada unidade é um episódio separado. Os caches das unidades não interagem: se elas decidissem em sequência no mesmo episódio, o "próximo estado" de uma decisão seria o cache de outra unidade, que o agente não vê, e a propriedade de Markov se perderia. Para avaliar um texto, rodamos todas as unidades e tiramos a média. As regras fixas também passam a decidir por unidade (o H2O por cabeça é o método original).

**Os modos são comparáveis entre si.** A atenção perdida é linear na matriz de atenção: aplicar a mesma decisão em todas as unidades e tirar a média dá exatamente a perda medida na matriz média. Ou seja, o modo geral equivale a aplicar uma decisão única nos 144 / 48 caches, e o que os modos mais finos acrescentam é só a liberdade de decidir diferente em cada um. No LLM real a comparação é direta: mesmo modelo, mesmos textos, mesmo orçamento em cada cache.

### DQN

Stable-Baselines3, com duas escolhas que vieram da primeira execução no Qwen, em que o DQN perdeu do H2O em todos os modos e, no modo geral, terminou pior que a rede antes do treino.

**Rede Q por candidato** (`src/kvcache_rl/rede.py`). A `MlpPolicy` recebe os 17 candidatos concatenados e precisa aprender a mesma regra 17 vezes, uma por posição. A rede por candidato aplica a mesma rede pequena a cada candidato (um embedding por candidato, a média deles como resumo do cache e o contexto de progresso e unidade) e devolve uma nota por candidato, como no Deep Sets. Ela consegue representar o H2O exatamente: treinada para imitá-lo, acerta mais de 95% das escolhas. Tem ~17 mil parâmetros. `--rede mlp` volta à versão anterior.

**Crédito por despejo.** Na versão anterior, a recompensa de cada passo era a atenção que o bloco lido daria a **todos** os despejados até ali, inclusive por decisões antigas que o agente não vê no estado. Agora cada decisão é cobrada pela atenção que o bloco que ela despejou receberia até o fim do texto. A soma do episódio é exatamente a mesma (há um teste); muda só a hora em que cada custo é cobrado, como na redistribuição de recompensa do RUDDER. O futuro entra só na recompensa, calculada depois do texto; a política decide só com o passado. `--credito por_passo` volta à versão anterior.

Hiperparâmetros em [`configs/dqn.yml`](configs/dqn.yml), no mesmo formato do `dqn.yml` da atividade de Atari, com uma entrada por modo:

| Parâmetro | `geral` | `por_camada` | `por_cabeca` | Por quê |
|---|---|---|---|---|
| `n_timesteps` | 200.000 | 300.000 | 500.000 | ~4.250, ~6.400 e ~10.600 episódios de 47 decisões |
| `policy` | `PorCandidato`, `oculta: 64` | igual | igual | uma nota por candidato, pesos compartilhados |
| `learning_starts` | 5.000 | 5.000 | 5.000 | episódios curtos; não precisa de 100 mil passos aleatórios |
| `exploration_fraction` | 0,3 | 0,3 | 0,3 | ε de 1,0 a 0,05 nos primeiros 30% do treino |
| `buffer_size` | 100k | 100k | 200k | |
| `learning_rate`, `train_freq`, `target_update_interval` | 1e-4, 4, 1.000 | iguais | iguais | como no Atari |

A política gulosa (sem exploração) é avaliada nos 50 textos de teste, em todas as unidades, a cada 5.000 / 10.000 / 25.000 passos. Treinamos com 3 sementes.

**O teto das features.** Num experimento de controle, a mesma rede foi treinada por supervisão com o custo real de despejar **cada** um dos 17 candidatos, o que é mais informação do que o DQN recebe. Mesmo assim ela só empatou com o TOVA (sintético: 0,030 contra 0,028; GPT-2: 0,097 contra 0,099). Features mais ricas (atenção em log, do último bloco, pico e tendência) não mudaram isso. Com a história de atenção que o agente vê, o TOVA já está perto do melhor possível; vencê-lo por margem grande exigiria informação que a atenção passada não traz.

### Métricas: o que medimos e por quê

A atenção perdida é um **proxy**: rápida de calcular, por isso serve para treinar, mas não diz o que acontece com o texto que o modelo prevê. Uma versão anterior também dava um bônus de `+10` por manter a agulha no cache; ele foi removido porque usava informação que a política não teria (onde está a agulha) e dominava o retorno (respondia por 97% do sinal). As métricas finais são medidas no LLM:

| Métrica | O que é | Onde |
|---|---|---|
| **KL por token** (principal) | KL(p_cheio ‖ p_podado) da distribuição do próximo token, nas previsões feitas depois que o cache enche | LLM real |
| **Aumento da perplexidade** | `exp(ΔNLL) − 1`, com ΔNLL = aumento médio da NLL por token com o cache podado, contra o cache completo | LLM real |
| **Senha recuperada** | a decodificação gulosa ainda produz a senha inteira; só nos textos em que o modelo acerta com o cache completo | LLM real |
| Atenção perdida | atenção que os blocos apagados receberiam com o cache completo | proxy (treino) |
| Proxy × real | correlação de Spearman entre atenção perdida e KL, **dentro de cada texto**, só entre as políticas competitivas (StreamingLLM, H2O, TOVA e o DQN); a aleatória e a janela deslizante são tão piores que inflariam a correlação | os dois |

**Como o LLM é avaliado com o cache podado.** Despejar um bloco do KV-cache equivale a proibir que as consultas seguintes o vejam. Então os despejos de uma política viram uma máscara de atenção — uma por camada, ou por cabeça de K/V — e um único forward com essa máscara dá exatamente o que o modelo calcularia apagando os blocos do cache durante a leitura (as posições dos tokens que ficam não mudam, como no H2O e no SnapKV). Ao contrário do proxy, aqui o softmax renormaliza: a atenção que iria para um bloco apagado se redistribui entre os que ficaram, como no modelo de verdade. O teste `test_mascara_igual_a_despejo_fisico` confere, no Qwen2 e no GPT-2, que os logits saem iguais aos de um KV-cache do qual os blocos foram de fato apagados.

**Comparação justa.** O DQN escolhe os pesos dele nos textos de treino, minimizando a atenção perdida. As regras com parâmetro recebem o mesmo tratamento: `avaliar_baselines.py` escolhe, numa grade e **nos textos de treino**, quantos blocos recentes o H2O protege (de 0 até o orçamento inteiro) e quantos blocos de sink o StreamingLLM mantém (1 a 4), pelo mesmo critério. O **TOVA** (Oren et al., 2024), que despeja o bloco com menos atenção recente, não tem parâmetro. Ajustá-las pela KL no modelo real daria a elas um sinal melhor que o do DQN, e a injustiça só trocaria de lado. O teste não escolhe nada; o parâmetro escolhido e a grade inteira ficam em `ajuste.json`.

As diferenças entre políticas são **pareadas** (mesmos textos) e vêm com intervalo de 95%: o script diz, para cada política, se ela é melhor, pior ou empata com o H2O e com o TOVA. A avaliação roda em float32 por padrão, porque em bfloat16 o arredondamento já mexe na NLL na terceira casa.

### Resultados no Qwen2.5-0.5B

50 textos de teste de 1.024 tokens, cache de 16 blocos (25% do texto), 3 sementes do DQN por modo. Com o cache completo, o modelo acerta a senha em todos os textos.

| Modo | KL do H2O ajustado | KL do TOVA | KL do DQN (média) | DQN contra H2O | DQN contra TOVA | Senha: DQN (regras: 0%) |
|---|---|---|---|---|---|---|
| Geral | 0,130 | 0,118 | 0,118 | 3 sementes melhores | 3 empates | 13% |
| Por camada | 0,114 | 0,112 | 0,111 | 1 melhor, 2 empates | 1 melhor, 2 empates | 1% |
| Por cabeça | 0,106 | 0,111 | 0,108 | 3 empates | 3 empates | 0% |

KL em nats por token; vereditos pareados, texto a texto, com IC de 95%.

- **O DQN ganha do H2O ajustado no modo geral** (−0,011 a −0,015 nats/token, as três sementes) e empata com o TOVA. Nos modos mais finos, os competidores empatam. Nenhuma semente perde de nenhuma regra.
- **Nenhuma regra fixa é a melhor nos três modos**: o TOVA vence no geral, e o H2O fica à frente por cabeça. O DQN acompanha a melhor em cada modo com o mesmo código, como previa o teto das features.
- **A primeira versão** (MLP, crédito por passo) perdia do H2O nas 9 sementes (KL 0,156 / 0,131 / 0,118). As regras deram os mesmos números nas duas rodadas, então a melhora é do agente.
- **O proxy mal separa as boas políticas**: o Spearman dentro do texto entre atenção perdida e KL é 0,44 / 0,36 / 0,34 entre as competitivas. No modo geral, o DQN tem a mesma atenção perdida do H2O (10,62%) e KL 10% menor.
- **Cautela:** o desenho final do DQN (rede por candidato, crédito por despejo) foi escolhido depois de ver a primeira rodada nos mesmos textos de teste. O modelo avaliado é o do fim do treino, não o melhor checkpoint.

### Como rodar

Requer Python 3.10+ (no Debian/Ubuntu: `sudo apt install python3-venv`).

```bash
make setup-llm              # cria .venv e instala tudo (PyTorch, SB3, transformers)
make teste                  # testes do ambiente e da gravação da atenção
make tudo-modos             # traces do GPT-2 -> três modos (baselines, DQN, avaliação no LLM) -> comparação
make tudo-modos FONTE=qwen  # o mesmo com o Qwen2.5-0.5B, em dados/qwen e resultados/qwen
```

Um modo só: `make tudo MODO=por_cabeca` (ou `make baselines treinar graficos MODO=por_camada` sem regerar os traces). Sem GPU ou sem internet: `make setup && make tudo-modos FONTE=sintetico`.

Etapa por etapa (troque `qwen` por `gpt2` para o outro modelo):

```bash
.venv/bin/python scripts/gerar_traces_llm.py --fonte qwen                   # dados/qwen/{treino,teste}
.venv/bin/python scripts/avaliar_baselines.py --fonte qwen --modo por_cabeca   # ajusta H2O/StreamingLLM no treino
.venv/bin/python scripts/treinar_dqn.py --fonte qwen --modo por_cabeca      # resultados/qwen/por_cabeca/dqn/seed{0,1,2}
.venv/bin/python scripts/graficos.py --fonte qwen --modo por_cabeca         # resultados/qwen/por_cabeca/figuras
.venv/bin/python scripts/avaliar_no_modelo.py --fonte qwen --modo por_cabeca  # perplexidade e senha no LLM
.venv/bin/python scripts/comparar_modos.py --fonte qwen                     # proxy, os três modos
.venv/bin/python scripts/graficos_modelo_real.py --fonte qwen               # LLM real, os três modos
```

Contextos mais longos: `--tokens 2048` ou `--tokens 4096` (o Qwen aceita até 32 mil). O custo da gravação cresce com o quadrado do número de tokens, e numa GPU de 6 GB o 4.096 ainda cabe. Para o episódio continuar do mesmo tamanho, aumente o bloco junto: `--tokens 4096 --bloco 64` mantém os 64 blocos e o orçamento de 16.

Tempo aproximado num notebook: traces de 2 a 10 min (GPU) ou 10 a 30 min (CPU); DQN por semente cerca de 2,5 min no `geral`, 4 min no `por_camada` e 12 min no `por_cabeca` com 144 caches (menos no Qwen, que tem 48); avaliação no LLM, por modo, alguns minutos na GPU (um forward por texto e política: 50 × 8). O `make tudo-modos` inteiro leva cerca de 1 hora e meia.

Os traces gravados antes desta versão não guardavam os tokens do texto e não servem para a avaliação no LLM: gere-os de novo (`make traces FONTE=qwen`).

### Saídas

| Arquivo | Conteúdo |
|---|---|
| `resultados/<fonte>/<modo>/baselines.csv` | retorno, atenção perdida e agulha (proxy) de cada política de referência, no teste |
| `resultados/<fonte>/<modo>/ajuste.json` | parâmetros do H2O e do StreamingLLM escolhidos nos textos de treino, com a grade inteira |
| `resultados/<fonte>/<modo>/dqn/seed*/curva.csv` | avaliação no teste ao longo do treino |
| `resultados/<fonte>/<modo>/dqn/seed*/final.json` | avaliação do modelo final (= último ponto da curva) |
| `resultados/<fonte>/<modo>/figuras/curva_aprendizado.png` | retorno no teste ao longo do treino, com as políticas de referência |
| `resultados/<fonte>/<modo>/figuras/comparacao_final.png` | atenção perdida e agulha no cache (proxy) por política |
| `resultados/<fonte>/<modo>/modelo_real.csv` | **por política: aumento da perplexidade e senha recuperada no LLM**, com IC de 95% |
| `resultados/<fonte>/<modo>/modelo_real_textos.csv` | o mesmo, texto a texto, com a atenção perdida ao lado |
| `resultados/<fonte>/<modo>/modelo_real_resumo.json` | correlação proxy × real e diferenças pareadas contra o H2O e o TOVA |
| `resultados/<fonte>/comparacao_modos_real.png` | **H2O, TOVA e DQN nos três modos, medidos no LLM** |
| `resultados/<fonte>/proxy_x_real.png` | atenção perdida × aumento da perplexidade, por texto e política |
| `resultados/<fonte>/comparacao_modos.png` e `.csv` | H2O, TOVA e DQN nos três modos, no proxy |

### Estrutura

```
configs/dqn.yml                  hiperparâmetros do DQN, um bloco por modo
src/kvcache_rl/
  env.py                         ambiente Gymnasium (KVCacheEnv) com os três modos
  traces.py                      formato dos traces (por bloco; modelo, camada e cabeça) e gerador sintético
  llm.py                         gravação da atenção por cabeça de K/V (GPT-2, Qwen), agulha e NLL
  modelo_real.py                 despejos de uma política -> máscara de atenção no LLM real; métricas
  baselines.py                   janela deslizante, StreamingLLM, H2O, TOVA, aleatória, ajuste e avaliação
  rede.py                        rede Q por candidato (pesos compartilhados) para o DQN do SB3
  config.py, visual.py           leitura do dqn.yml e estilo dos gráficos
scripts/                         gerar traces, avaliar (proxy e LLM), treinar, gráficos, comparar modos
tests/                           ambiente, gravação da atenção e equivalência máscara = despejo de verdade
```

## Entregas

- [x] **Desafio 1** — apresentação do problema: ambiente/estado, ações e recompensas — 29/10/2026 ([slides](apresentacoes/desafio1/))
- [x] **Desafio 2** — DQN no problema e gráficos dos resultados — 12/11/2026

## Referências

- Xiao et al. *Efficient Streaming Language Models with Attention Sinks* (StreamingLLM). ICLR 2024. [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
- Zhang et al. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models*. NeurIPS 2023. [arXiv:2306.14048](https://arxiv.org/abs/2306.14048)
- Li et al. *SnapKV: LLM Knows What You are Looking for Before Generation*. 2024. [arXiv:2404.14469](https://arxiv.org/abs/2404.14469)
- Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM). SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- Wu et al. *Retrieval Head Mechanistically Explains Long-Context Factuality*. 2024. [arXiv:2404.15574](https://arxiv.org/abs/2404.15574)
- Xiao et al. *DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads*. 2024. [arXiv:2410.10819](https://arxiv.org/abs/2410.10819)
- Fu et al. *Not All Heads Matter: A Head-Level KV Cache Compression Method with Integrated Retrieval and Reasoning* (HeadKV). 2024. [arXiv:2410.19258](https://arxiv.org/abs/2410.19258)
- Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*. EMNLP 2023. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- Qwen Team. *Qwen2.5 Technical Report*. 2024. [arXiv:2412.15115](https://arxiv.org/abs/2412.15115)
- Mnih et al. *Human-level control through deep reinforcement learning* (DQN). Nature, 2015.
- Oren et al. *Transformers are Multi-State RNNs* (TOVA). 2024. [arXiv:2401.06104](https://arxiv.org/abs/2401.06104)
- Zaheer et al. *Deep Sets*. NeurIPS 2017. [arXiv:1703.06114](https://arxiv.org/abs/1703.06114)
- Arjona-Medina et al. *RUDDER: Return Decomposition for Delayed Rewards*. NeurIPS 2019. [arXiv:1806.07857](https://arxiv.org/abs/1806.07857)
- Merity et al. *Pointer Sentinel Mixture Models* (WikiText). 2016. [arXiv:1609.07843](https://arxiv.org/abs/1609.07843)
- Sutton & Barto. *Reinforcement Learning: An Introduction*, 2ª ed., 2018.

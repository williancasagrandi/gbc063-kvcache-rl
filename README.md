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

## Plano ao longo da disciplina

| Etapa | Conteúdo da disciplina |
|---|---|
| Desafio 1: contextualização do problema (este repositório) | MDP, propriedade de Markov, POMDP |
| Ambiente simulado e políticas de referência | MDP, avaliação de política |
| Primeiro agente com DQN | controle sem modelo e aproximação de função |
| Novas abordagens conforme o curso avança | gradiente de política, PPO, imitação, entre outros |
| Validação num LLM pequeno (GPT-2) | — |

## Entregas

- [x] **Desafio 1** — apresentação do problema: ambiente/estado, ações e recompensas — 29/10/2026 ([slides](apresentacoes/desafio1/))

## Referências

- Xiao et al. *Efficient Streaming Language Models with Attention Sinks* (StreamingLLM). ICLR 2024. [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
- Zhang et al. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models*. NeurIPS 2023. [arXiv:2306.14048](https://arxiv.org/abs/2306.14048)
- Li et al. *SnapKV: LLM Knows What You are Looking for Before Generation*. 2024. [arXiv:2404.14469](https://arxiv.org/abs/2404.14469)
- Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM). SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- Sutton & Barto. *Reinforcement Learning: An Introduction*, 2ª ed., 2018.

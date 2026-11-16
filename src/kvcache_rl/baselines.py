"""Políticas de referência e avaliação. Uma política recebe (env, obs, info)
e devolve o índice do candidato a despejar.

Ajuste justo: o DQN é treinado nos textos de treino para minimizar a atenção
perdida. As regras com parâmetro (H2O e StreamingLLM) recebem o mesmo tratamento:
o parâmetro é escolhido numa grade, nos textos de treino, pelo mesmo critério
(ajustar). Os textos de teste não são usados para escolher nada.
"""

from __future__ import annotations

import itertools
import json
from functools import partial
from pathlib import Path

import numpy as np


def aleatoria(env, obs, info):
    return int(env.np_random.integers(env.n_cand))


def janela_deslizante(env, obs, info):
    """Despeja sempre o bloco mais antigo."""
    return int(np.argmin(info["candidatos"]))


def streaming_llm(env, obs, info, n_sink: int | None = None):
    """Mantém os `n_sink` primeiros blocos (os attention sinks) e despeja o mais antigo
    entre os demais. Sem `n_sink`, usa o do ambiente (1 bloco = 16 tokens)."""
    n = env.n_sink if n_sink is None else n_sink
    cand = info["candidatos"]
    idade = np.where(cand >= n, cand, np.iinfo(np.int64).max)
    return int(np.argmin(idade))


def h2o(env, obs, info, n_recentes: int | None = None):
    """Heavy-Hitter Oracle: protege os `n_recentes` blocos mais recentes e despeja,
    entre os outros, o que recebeu menos atenção acumulada até agora. Sem `n_recentes`,
    metade do orçamento, como no artigo (metade recentes, metade heavy hitters)."""
    n = env.B // 2 if n_recentes is None else n_recentes
    score = info["atencao_acumulada"].astype(np.float64).copy()
    if n > 0:  # score[-0:] seria o vetor inteiro
        score[-n:] = np.inf
    return int(np.argmin(score))


def tova(env, obs, info):
    """TOVA (Oren et al., 2024): despeja o bloco que recebeu menos atenção das consultas mais
    recentes. Aqui, a média dos últimos `janela_recente` blocos do ambiente (2 blocos = 32 tokens),
    a mesma feature que o DQN vê. Não tem parâmetro para ajustar."""
    return int(np.argmin(info["atencao_recente"]))


BASELINES = {
    "aleatoria": aleatoria,
    "janela_deslizante": janela_deslizante,
    "streaming_llm": streaming_llm,
    "h2o": h2o,
    "tova": tova,
}

NOMES = {
    "aleatoria": "Aleatória",
    "janela_deslizante": "Janela deslizante",
    "streaming_llm": "StreamingLLM",
    "h2o": "H2O",
    "tova": "TOVA",
    "dqn": "DQN",
}

# Grades de ajuste (em blocos). Uma lista fixa ou uma função do ambiente.
# StreamingLLM: 1 a 4 blocos de sink (o artigo usa 4 tokens, que cabem no 1º bloco;
#   0 seria a janela deslizante, que já é outra política).
# H2O: o artigo divide o orçamento entre recentes e "heavy hitters"; aqui o parâmetro
#   é quantos recentes proteger, e a grade cobre TODOS os valores (0 = só heavy hitters,
#   orçamento = só recentes, igual à janela deslizante), para o ótimo nunca ficar de fora.
PARAMETROS = {
    "streaming_llm": {"n_sink": [1, 2, 3, 4]},
    "h2o": {"n_recentes": lambda env: list(range(env.B + 1))},
}


def grade_de(nome: str, env) -> dict:
    """A grade de `nome` com os valores resolvidos para este ambiente."""
    return {k: (v(env) if callable(v) else list(v)) for k, v in PARAMETROS[nome].items()}


def com_parametros(nome: str, params: dict | None = None):
    """A política `nome` com os parâmetros dados (sem parâmetros: a própria função)."""
    return partial(BASELINES[nome], **params) if params else BASELINES[nome]


def ajustar(env_treino, nome: str) -> dict:
    """Escolhe os parâmetros de `nome` que minimizam a atenção perdida nos textos de treino.

    É o mesmo critério e os mesmos dados que o DQN usa no treino. Empates ficam com
    o primeiro da grade. Devolve {"melhor": {...}, "grade": [{params..., atencao_perdida}]}.
    """
    grade = grade_de(nome, env_treino)
    resultados = []
    for valores in itertools.product(*grade.values()):
        params = dict(zip(grade, valores))
        r = avaliar(env_treino, com_parametros(nome, params))
        resultados.append({**params, "atencao_perdida": r["atencao_perdida"]})
    melhor = min(resultados, key=lambda r: r["atencao_perdida"])
    return {"melhor": {k: melhor[k] for k in grade}, "grade": resultados}


def politicas_ajustadas(caminho_ajuste: str | Path) -> tuple[dict, bool]:
    """Políticas de referência com os parâmetros de ajuste.json, se existir.
    Devolve (nome -> política, se o ajuste foi encontrado)."""
    caminho = Path(caminho_ajuste)
    ajuste = json.loads(caminho.read_text()) if caminho.exists() else {}
    return {n: com_parametros(n, ajuste.get(n, {}).get("melhor")) for n in BASELINES}, caminho.exists()


def avaliar(env, politica, seed: int = 0) -> dict:
    """Roda a política uma vez em cada trace do env, em cada unidade (o modelo todo,
    cada camada ou cada cabeça, conforme o modo), e devolve as médias.

    atenção perdida: média sobre todas as unidades e textos.
    agulha: por texto, a fração da atenção que a pergunta dá à agulha que foi
    preservada (unidades ponderadas pelo quanto consultam a agulha); média nos textos.
    """
    retornos, perdas = [], []
    agulha_texto: dict[int, float] = {}
    for n, opcoes in enumerate(env.episodios_avaliacao()):
        obs, info = env.reset(seed=seed if n == 0 else None, options=opcoes)
        G, fim = 0.0, False
        while not fim:
            obs, r, term, trunc, info = env.step(politica(env, obs, info))
            G += r
            fim = term or trunc
        retornos.append(G)
        perdas.append(info["atencao_perdida_media"])
        if "agulha_mantida" in info:
            i = opcoes["trace"]
            agulha_texto[i] = agulha_texto.get(i, 0.0) + info["peso_agulha"] * info["agulha_mantida"]
    return {
        "retorno": float(np.mean(retornos)),
        "retorno_dp": float(np.std(retornos)),
        "atencao_perdida": float(np.mean(perdas)),
        "agulha": float(np.mean(list(agulha_texto.values()))) if agulha_texto else float("nan"),
    }


def avaliar_dqn(env, modelo, lote: int = 512) -> dict:
    """Mesmo resultado de avaliar(env, politica_sb3(modelo)), mas roda até `lote`
    episódios em paralelo e chama a rede Q uma vez por passo para todos eles.
    Faz diferença no modo por_cabeca (7.200 episódios por avaliação)."""
    import copy

    import torch

    q_net, dispositivo = modelo.policy.q_net, modelo.device
    episodios = env.episodios_avaliacao()
    retornos, perdas = [], []
    agulha_texto: dict[int, float] = {}
    pool = [copy.copy(env) for _ in range(min(lote, len(episodios)))]
    for ini in range(0, len(episodios), lote):
        bloco_ep = episodios[ini:ini + lote]
        envs = pool[:len(bloco_ep)]
        obs = np.stack([e.reset(options=o)[0] for e, o in zip(envs, bloco_ep)])
        G = np.zeros(len(envs))
        ativo = np.ones(len(envs), dtype=bool)
        while ativo.any():
            with torch.no_grad():
                q = q_net(torch.as_tensor(obs, dtype=torch.float32, device=dispositivo))
            acoes = q.argmax(dim=1).cpu().numpy()
            for n in np.flatnonzero(ativo):
                o, r, term, trunc, info = envs[n].step(int(acoes[n]))
                obs[n], G[n] = o, G[n] + r
                if term or trunc:
                    ativo[n] = False
                    perdas.append(info["atencao_perdida_media"])
                    if "agulha_mantida" in info:
                        i = bloco_ep[n]["trace"]
                        agulha_texto[i] = agulha_texto.get(i, 0.0) + info["peso_agulha"] * info["agulha_mantida"]
        retornos.extend(G.tolist())
    return {
        "retorno": float(np.mean(retornos)),
        "retorno_dp": float(np.std(retornos)),
        "atencao_perdida": float(np.mean(perdas)),
        "agulha": float(np.mean(list(agulha_texto.values()))) if agulha_texto else float("nan"),
    }


def politica_sb3(modelo):
    """Adapta um modelo do Stable-Baselines3 (ação gulosa, sem exploração).

    Para o DQN chama a rede Q direto (argmax de Q), bem mais rápido que predict()
    nas avaliações com muitas decisões; o resultado é o mesmo.
    """
    q_net = getattr(getattr(modelo, "policy", None), "q_net", None)
    if q_net is not None:
        import torch

        dispositivo = modelo.device

        def _pol(env, obs, info):
            with torch.no_grad():
                q = q_net(torch.as_tensor(obs, dtype=torch.float32, device=dispositivo).unsqueeze(0))
            return int(q.argmax(dim=1).item())
        return _pol

    def _pol(env, obs, info):
        a, _ = modelo.predict(obs, deterministic=True)
        return int(a)
    return _pol

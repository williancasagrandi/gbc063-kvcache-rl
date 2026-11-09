"""Políticas de referência e avaliação. Uma política recebe (env, obs, info)
e devolve o índice do candidato a despejar."""

from __future__ import annotations

import numpy as np


def aleatoria(env, obs, info):
    return int(env.np_random.integers(env.n_cand))


def janela_deslizante(env, obs, info):
    """Despeja sempre o bloco mais antigo."""
    return int(np.argmin(info["candidatos"]))


def streaming_llm(env, obs, info):
    """Mantém os sinks e despeja o mais antigo entre os demais."""
    cand = info["candidatos"]
    idade = np.where(cand >= env.n_sink, cand, np.iinfo(np.int64).max)
    return int(np.argmin(idade))


def h2o(env, obs, info, n_recentes: int = 4):
    """Heavy-Hitter Oracle: protege os n mais recentes e despeja o menos atendido."""
    score = info["atencao_acumulada"].astype(np.float64).copy()
    score[-n_recentes:] = np.inf
    return int(np.argmin(score))


def oraculo(env, obs, info):
    """Despeja o bloco que receberá menos atenção no futuro e nunca a agulha.
    Não é implementável na prática (usa o futuro): é o limite superior."""
    futura = env.massa_futura()
    if env.trace.agulha >= 0:
        futura = futura.copy()
        futura[info["candidatos"] == env.trace.agulha] = np.inf
    return int(np.argmin(futura))


BASELINES = {
    "aleatoria": aleatoria,
    "janela_deslizante": janela_deslizante,
    "streaming_llm": streaming_llm,
    "h2o": h2o,
    "oraculo": oraculo,
}

NOMES = {
    "aleatoria": "Aleatória",
    "janela_deslizante": "Janela deslizante",
    "streaming_llm": "StreamingLLM",
    "h2o": "H2O",
    "oraculo": "Oráculo",
    "dqn": "DQN",
}


def avaliar(env, politica, seed: int = 0) -> dict:
    """Roda a política uma vez em cada trace do env (e em cada camada, no modo
    por_camada) e devolve as médias."""
    retornos, perdas, agulhas = [], [], []
    for i, opcoes in enumerate(env.episodios_avaliacao()):
        obs, info = env.reset(seed=seed if i == 0 else None, options=opcoes)
        G, fim = 0.0, False
        while not fim:
            obs, r, term, trunc, info = env.step(politica(env, obs, info))
            G += r
            fim = term or trunc
        retornos.append(G)
        perdas.append(info["atencao_perdida_media"])
        if info.get("agulha_mantida") is not None:
            agulhas.append(float(info["agulha_mantida"]))
    return {
        "retorno": float(np.mean(retornos)),
        "retorno_dp": float(np.std(retornos)),
        "atencao_perdida": float(np.mean(perdas)),
        "agulha": float(np.mean(agulhas)) if agulhas else float("nan"),
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

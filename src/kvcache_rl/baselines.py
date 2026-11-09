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
    """Despeja o bloco que receberá menos atenção no resto do texto.

    Não é implementável na prática (olha o futuro) e serve de referência. É guloso:
    decide um passo de cada vez, então é um limite muito bom, mas não o ótimo
    garantido. Se a recompensa tiver bônus de agulha, também nunca despeja a agulha.
    """
    futura = env.massa_futura()
    if env.bonus_agulha > 0 and env.trace.agulha >= 0:
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

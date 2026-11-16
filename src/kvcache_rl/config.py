"""Leitura do configs/dqn.yml (uma entrada por modo do ambiente)."""

from __future__ import annotations

import yaml

CHAVES = {"geral": "KVCache-v0", "por_camada": "KVCachePorCamada-v0", "por_cabeca": "KVCachePorCabeca-v0"}
NOMES_MODOS = {
    "geral": "Geral (média do modelo)",
    "por_camada": "Decisão por camada",
    "por_cabeca": "Decisão por cabeça de K/V",
}


def carregar_config(caminho: str, modo: str) -> dict:
    """Devolve dict com: dqn (kwargs do DQN), env (kwargs do ambiente), passos, eval_freq,
    rede ("por_candidato" ou "mlp") e mlp_net_arch (camadas da MLP, para --rede mlp)."""
    if modo not in CHAVES:
        raise SystemExit(f"modo desconhecido: {modo} (use {', '.join(CHAVES)})")
    with open(caminho) as f:
        cfg = dict(yaml.safe_load(f)[CHAVES[modo]])
    passos = int(float(cfg.pop("n_timesteps")))
    eval_freq = int(cfg.pop("eval_freq", 5000))
    env = dict(cfg.pop("env_kwargs", {}))
    policy = cfg.pop("policy")
    rede = "por_candidato" if policy == "PorCandidato" else "mlp"
    mlp_net_arch = cfg.pop("mlp_net_arch", None) or (cfg.get("policy_kwargs") or {}).get("net_arch", [128, 128])
    for k, v in cfg.items():  # yml pode trazer números como texto ("1e-4")
        if isinstance(v, str):
            try:
                cfg[k] = float(v)
            except ValueError:
                pass
    return {"policy": policy, "rede": rede, "mlp_net_arch": mlp_net_arch, "dqn": cfg, "env": env,
            "passos": passos, "eval_freq": eval_freq}

"""Leitura do configs/dqn.yml (uma entrada por modo do ambiente)."""

from __future__ import annotations

import yaml

CHAVES = {"geral": "KVCache-v0", "por_camada": "KVCachePorCamada-v0", "por_cabeca": "KVCachePorCabeca-v0"}
NOMES_MODOS = {
    "geral": "Geral (média do modelo)",
    "por_camada": "Decisão por camada",
    "por_cabeca": "Decisão por cabeça",
}


def carregar_config(caminho: str, modo: str) -> dict:
    """Devolve dict com: dqn (kwargs do DQN), env (kwargs do ambiente), passos, eval_freq."""
    if modo not in CHAVES:
        raise SystemExit(f"modo desconhecido: {modo} (use {', '.join(CHAVES)})")
    with open(caminho) as f:
        cfg = dict(yaml.safe_load(f)[CHAVES[modo]])
    passos = int(float(cfg.pop("n_timesteps")))
    eval_freq = int(cfg.pop("eval_freq", 5000))
    env = dict(cfg.pop("env_kwargs", {}))
    policy = cfg.pop("policy")
    for k, v in cfg.items():  # yml pode trazer números como texto ("1e-4")
        if isinstance(v, str):
            try:
                cfg[k] = float(v)
            except ValueError:
                pass
    return {"policy": policy, "dqn": cfg, "env": env, "passos": passos, "eval_freq": eval_freq}

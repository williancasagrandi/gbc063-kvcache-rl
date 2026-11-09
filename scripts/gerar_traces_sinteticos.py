"""Gera traces sintéticos de treino e teste (não precisa de GPU nem de internet).

    python scripts/gerar_traces_sinteticos.py

Saída: dados/sintetico/treino/*.npz e dados/sintetico/teste/*.npz.
"""

import argparse
from pathlib import Path

import numpy as np

from kvcache_rl.traces import gerar_sintetico


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-treino", type=int, default=200)
    p.add_argument("--n-teste", type=int, default=50)
    p.add_argument("--blocos", type=int, default=64)
    p.add_argument("--bloco", type=int, default=16)
    p.add_argument("--saida", default="dados/sintetico")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    for parte, n in (("treino", a.n_treino), ("teste", a.n_teste)):
        for i in range(n):
            tr = gerar_sintetico(a.blocos, a.bloco, rng=rng, nome=f"{parte}_{i:04d}")
            tr.salvar(Path(a.saida) / parte / f"{tr.nome}.npz")
    print(f"{a.n_treino} traces de treino e {a.n_teste} de teste em {a.saida}/")


if __name__ == "__main__":
    main()

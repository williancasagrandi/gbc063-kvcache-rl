"""Avalia as políticas de referência nos traces de teste.

    python scripts/avaliar_baselines.py --fonte gpt2

Saída: resultados/<fonte>/baselines.csv
"""

import argparse
import csv
from pathlib import Path

import yaml

from kvcache_rl import KVCacheEnv, carregar_pasta
from kvcache_rl.baselines import BASELINES, NOMES, avaliar


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fonte", default="gpt2")
    p.add_argument("--config", default="configs/dqn.yml")
    a = p.parse_args()

    env_kwargs = yaml.safe_load(open(a.config))["KVCache-v0"].get("env_kwargs", {})
    env = KVCacheEnv(carregar_pasta(f"dados/{a.fonte}/teste"), **env_kwargs)

    linhas = []
    print(f"{'política':<18} {'retorno':>9} {'atenção perdida':>16} {'agulha':>8}")
    for chave, pol in BASELINES.items():
        r = avaliar(env, pol)
        linhas.append({"politica": chave, **r})
        print(f"{NOMES[chave]:<18} {r['retorno']:>9.2f} {r['atencao_perdida']:>16.3f} {r['agulha']:>8.2f}")

    saida = Path(f"resultados/{a.fonte}/baselines.csv")
    saida.parent.mkdir(parents=True, exist_ok=True)
    with open(saida, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    print(f"\nsalvo em {saida}")


if __name__ == "__main__":
    main()

"""Ajusta as regras fixas nos textos de treino e as avalia nos de teste (proxy).

    python scripts/avaliar_baselines.py --fonte gpt2 --modo geral
    python scripts/avaliar_baselines.py --fonte qwen --modo por_cabeca --n-ajuste 100

Ajuste justo: o DQN escolhe os pesos dele nos textos de treino, minimizando a
atenção perdida. O H2O (quantos blocos recentes proteger) e o StreamingLLM
(quantos blocos de sink manter) escolhem o parâmetro deles do mesmo jeito: numa
grade, nos textos de treino, pela atenção perdida. O teste não escolhe nada.

Nos modos por_camada e por_cabeca as regras também decidem por unidade (ex.: o
H2O usa a atenção de cada cabeça, como no método original), e o ajuste é por modo.

Saída em resultados/<fonte>/<modo>/:
    baselines.csv   retorno, atenção perdida e agulha (proxy) de cada política no teste
    ajuste.json     parâmetro escolhido e a grade inteira, para cada regra ajustada
"""

import argparse
import csv
import json
from pathlib import Path

from kvcache_rl import KVCacheEnv, carregar_pasta
from kvcache_rl.baselines import BASELINES, NOMES, PARAMETROS, ajustar, avaliar, com_parametros
from kvcache_rl.config import CHAVES, carregar_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fonte", default="gpt2")
    p.add_argument("--modo", default="geral", choices=list(CHAVES))
    p.add_argument("--config", default="configs/dqn.yml")
    p.add_argument("--n-ajuste", type=int, default=50,
                   help="quantos textos de treino usar no ajuste (o modo por cabeça é lento com todos)")
    p.add_argument("--sem-ajuste", action="store_true", help="usa os parâmetros padrão das regras")
    a = p.parse_args()

    cfg = carregar_config(a.config, a.modo)
    base = Path(f"resultados/{a.fonte}/{a.modo}")
    base.mkdir(parents=True, exist_ok=True)

    ajuste = {}
    if not a.sem_ajuste:
        treino = carregar_pasta(f"dados/{a.fonte}/treino")[: a.n_ajuste]
        env_treino = KVCacheEnv(treino, **cfg["env"])
        print(f"modo {a.modo}: ajustando nas {len(treino)} primeiras amostras de treino (atenção perdida)")
        for nome in PARAMETROS:
            ajuste[nome] = ajustar(env_treino, nome)
            grade = "  ".join(f"{list(g.values())[0]}: {g['atencao_perdida'] * 100:.2f}%" for g in ajuste[nome]["grade"])
            print(f"  {NOMES[nome]:<13} {grade}   ->  {ajuste[nome]['melhor']}")
        (base / "ajuste.json").write_text(json.dumps(ajuste, indent=2))

    env = KVCacheEnv(carregar_pasta(f"dados/{a.fonte}/teste"), **cfg["env"])
    linhas = []
    print(f"\nteste ({len(env.traces)} textos)")
    print(f"{'política':<18} {'retorno':>9} {'atenção perdida':>16} {'agulha':>8}")
    for chave in BASELINES:
        r = avaliar(env, com_parametros(chave, ajuste.get(chave, {}).get("melhor")))
        linhas.append({"politica": chave, **r})
        print(f"{NOMES[chave]:<18} {r['retorno']:>9.2f} {r['atencao_perdida']:>16.3f} {r['agulha']:>8.2f}")

    with open(base / "baselines.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    print(f"\nsalvo em {base}/baselines.csv" + (" e ajuste.json" if ajuste else ""))


if __name__ == "__main__":
    main()

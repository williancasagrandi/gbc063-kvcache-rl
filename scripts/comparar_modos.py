"""Compara os três modos (geral, grupos, por_camada) num gráfico e numa tabela.

    python scripts/comparar_modos.py --fonte gpt2

Para cada modo mostra o H2O (melhor regra fixa), o DQN e o oráculo, em
atenção perdida e agulha mantida. Usa os resultados já salvos por
avaliar_baselines.py e treinar_dqn.py; modos sem resultado são pulados.

Saída: resultados/<fonte>/comparacao_modos.png e comparacao_modos.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from kvcache_rl.config import CHAVES, NOMES_MODOS
from kvcache_rl.visual import FONTES, TEMAS, estilo, carregar_resultados, num, preparar_matplotlib, resumo_dqn

plt = preparar_matplotlib()


def coletar(fonte: str) -> list[dict]:
    linhas = []
    for modo in CHAVES:
        try:
            baselines, _, finais = carregar_resultados(fonte, modo)
        except SystemExit:
            continue
        if not finais:
            continue
        linha = {"modo": modo}
        for chave in ("retorno", "atencao_perdida", "agulha"):
            linha[f"h2o_{chave}"] = baselines["h2o"][chave]
            linha[f"oraculo_{chave}"] = baselines["oraculo"][chave]
            m, lo, hi = resumo_dqn(finais, chave)
            linha[f"dqn_{chave}"], linha[f"dqn_{chave}_min"], linha[f"dqn_{chave}_max"] = m, lo, hi
        linha["sementes"] = len(finais)
        linhas.append(linha)
    if not linhas:
        raise SystemExit(f"nenhum modo com resultados em resultados/{fonte}/")
    return linhas


def grafico(linhas: list[dict], t: dict, saida: Path, fonte: str):
    fig, eixos = plt.subplots(1, 2, figsize=(12, 5.4), dpi=200, sharey=True)
    fig.patch.set_facecolor(t["fundo"])
    n = len(linhas)
    centros = np.arange(n)[::-1] * 1.0
    alt, desloc = 0.24, 0.27  # altura das barras e distância entre elas no grupo
    series = [("h2o", "H2O", -desloc), ("dqn", "DQN", 0.0), ("oraculo", "Oráculo", desloc)]

    paineis = [(eixos[0], "atencao_perdida", "Atenção perdida por token (%) — menor é melhor", 1),
               (eixos[1], "agulha", "Agulha mantida até o fim (%) — maior é melhor", 0)]
    for ax, chave, titulo, casas in paineis:
        estilo(ax, t)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=t["grade"], linewidth=0.8)
        vmax = max(max(l[f"h2o_{chave}"], l[f"dqn_{chave}_max"], l[f"oraculo_{chave}"]) for l in linhas) * 100
        for c, l in zip(centros, linhas):
            for pol, _, d in series:
                yi = c - d  # H2O em cima, oráculo embaixo
                v = l[f"{pol}_{chave}"] * 100
                if pol == "oraculo":
                    ax.barh(yi, v, height=alt, color="none", edgecolor=t["referencia"], linewidth=1.1,
                            linestyle=(0, (4, 3)))
                else:
                    ax.barh(yi, v, height=alt, color=t["destaque"] if pol == "dqn" else t["barra"])
                fim = v
                if pol == "dqn" and l["sementes"] > 1:
                    lo, hi = l[f"dqn_{chave}_min"] * 100, l[f"dqn_{chave}_max"] * 100
                    if hi > lo:
                        ax.plot([lo, hi], [yi, yi], color=t["texto"], linewidth=1.3)
                        for x in (lo, hi):
                            ax.plot([x, x], [yi - 0.07, yi + 0.07], color=t["texto"], linewidth=1.3)
                    fim = max(v, hi)
                ax.text(fim + vmax * 0.02, yi, num(v, casas) + "%", va="center", fontsize=11,
                        color=t["texto"] if pol != "oraculo" else t["secundario"],
                        fontweight="bold" if pol == "dqn" else "normal")
        ax.set_xlim(0, vmax * 1.2 if vmax > 0 else 1)
        if casas == 0:
            ax.set_xticks([x for x in (0, 20, 40, 60, 80, 100) if x <= vmax * 1.2])
        ax.set_title(titulo, color=t["texto"], fontsize=13, loc="left", pad=12)

    eixos[0].set_yticks(centros)
    eixos[0].set_yticklabels([NOMES_MODOS[l["modo"]].replace(" (", "\n(").replace(" de camadas", "\nde camadas")
                              for l in linhas], fontsize=12, color=t["texto"])

    # legenda (3 séries): quadrados na mesma cor das barras
    from matplotlib.patches import Patch
    itens = [Patch(facecolor=t["barra"], label="H2O (melhor regra fixa)"),
             Patch(facecolor=t["destaque"], label="DQN (traço = mín–máx das sementes)"),
             Patch(facecolor="none", edgecolor=t["referencia"], linestyle=(0, (4, 3)), label="Oráculo (vê o futuro)")]
    leg = fig.legend(handles=itens, loc="lower left", bbox_to_anchor=(0.2, 0.0), ncol=3, frameon=False,
                     fontsize=11, labelcolor=t["secundario"], handlelength=1.2, columnspacing=2.0)
    for h in leg.legend_handles:
        h.set_height(9)
    fig.text(0.2, 0.95, f"Mesmo orçamento de memória por camada nos três modos · textos de teste {FONTES.get(fonte, fonte)}"
             " · no modo por camada, agulha = fração das camadas que a mantiveram",
             color=t["secundario"], fontsize=10.5)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.86, bottom=0.14, wspace=0.12)
    fig.savefig(saida, facecolor=t["fundo"])
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fonte", default="gpt2")
    p.add_argument("--tema", choices=list(TEMAS), default="escuro")
    a = p.parse_args()

    linhas = coletar(a.fonte)
    print(f"{'modo':<32} {'':>8} {'retorno':>9} {'atenção perdida':>16} {'agulha':>8}")
    for l in linhas:
        for pol in ("h2o", "dqn", "oraculo"):
            print(f"{NOMES_MODOS[l['modo']] if pol == 'h2o' else '':<32} {pol:>8} {l[f'{pol}_retorno']:>9.2f} "
                  f"{l[f'{pol}_atencao_perdida']:>16.3f} {l[f'{pol}_agulha']:>8.2f}")

    base = Path(f"resultados/{a.fonte}")
    with open(base / "comparacao_modos.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    sufixo = "" if a.tema == "escuro" else "_claro"
    grafico(linhas, TEMAS[a.tema], base / f"comparacao_modos{sufixo}.png", a.fonte)
    print(f"\nsalvo em {base}/comparacao_modos{sufixo}.png e comparacao_modos.csv")


if __name__ == "__main__":
    main()

"""Estilo comum dos gráficos e leitura dos resultados salvos pelos scripts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

TEMAS = {
    # escuro = mesmo fundo dos slides
    "escuro": dict(fundo="#12161D", texto="#F1F5F9", secundario="#9AA7B8", grade="#262D3A",
                   eixo="#5B6778", barra="#5B6778", destaque="#C98500", referencia="#9AA7B8"),
    "claro": dict(fundo="#FFFFFF", texto="#0B0B0B", secundario="#52514E", grade="#E9E8E4",
                  eixo="#8A8880", barra="#B4B2AA", destaque="#2A78D6", referencia="#52514E"),
}
FONTES = {"gpt2": "GPT-2", "qwen": "Qwen2.5-0.5B", "sintetico": "sintéticos"}


def num(v: float, casas: int = 1) -> str:
    """Número no formato brasileiro, com sinal de menos tipográfico."""
    return f"{v:.{casas}f}".replace(".", ",").replace("-", "−")


def preparar_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    instaladas = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = next((f for f in ("Calibri", "Carlito") if f in instaladas), "DejaVu Sans")
    return plt


def estilo(ax, t):
    ax.set_facecolor(t["fundo"])
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(t["eixo"])
    ax.tick_params(colors=t["secundario"], labelsize=11, length=0)
    ax.grid(axis="y", color=t["grade"], linewidth=0.8)
    ax.set_axisbelow(True)
    from matplotlib.ticker import FuncFormatter

    formato = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "\u2212"))
    ax.xaxis.set_major_formatter(formato)
    ax.yaxis.set_major_formatter(formato)


def ler_csv(caminho: Path) -> list[dict]:
    with open(caminho) as f:
        return [{k: (v if k == "politica" else float(v)) for k, v in linha.items()} for linha in csv.DictReader(f)]


def carregar_resultados(fonte: str, modo: str):
    """baselines (dict por política), curvas (lista por semente), finais (lista por semente)."""
    base = Path(f"resultados/{fonte}/{modo}")
    if not (base / "baselines.csv").exists():
        raise SystemExit(f"faltou {base / 'baselines.csv'}: rode scripts/avaliar_baselines.py --modo {modo}")
    baselines = {l["politica"]: l for l in ler_csv(base / "baselines.csv")}
    pastas = sorted((base / "dqn").glob("seed*"))
    curvas = [ler_csv(p / "curva.csv") for p in pastas if (p / "curva.csv").exists()]
    finais = [json.loads((p / "final.json").read_text()) for p in pastas if (p / "final.json").exists()]
    return baselines, curvas, finais


def resumo_dqn(finais: list[dict], chave: str) -> tuple[float, float, float]:
    """média, mínimo e máximo entre as sementes."""
    v = np.array([f[chave] for f in finais])
    return float(v.mean()), float(v.min()), float(v.max())


def grafico_modos(plt, linhas: list[dict], series: list[tuple[str, str]], paineis: list[tuple],
                  t: dict, saida, subtitulo: str, legenda_traco: str):
    """Barras horizontais agrupadas: um grupo por modo, uma barra por política, dois painéis.

    linhas:  [{"modo": ..., "<pol>_<chave>": valor, "<pol>_<chave>_lo"/"_hi": traço (opcional)}]
    series:  [(pol, rótulo)], de cima para baixo; "dqn" é destacada, "oraculo" só contorno
    paineis: [(chave, título, casas decimais, escala)], escala 100 para mostrar em %
    """
    from matplotlib.patches import Patch

    from .config import NOMES_MODOS

    fig, eixos = plt.subplots(1, len(paineis), figsize=(12, 5.4), dpi=200, sharey=True)
    fig.patch.set_facecolor(t["fundo"])
    n, k = len(linhas), len(series)
    passo = 0.8 / k  # espaço de cada barra dentro do grupo
    alt = passo * 0.88
    centros = np.arange(n)[::-1] * 1.0
    desloc = [(k - 1) / 2 * passo - j * passo for j in range(k)]  # primeira série em cima

    for ax, (chave, titulo, casas, escala) in zip(np.atleast_1d(eixos), paineis):
        estilo(ax, t)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=t["grade"], linewidth=0.8)
        vals = [v * escala for l in linhas for pol, _ in series
                for v in (l.get(f"{pol}_{chave}"), l.get(f"{pol}_{chave}_lo"), l.get(f"{pol}_{chave}_hi"))
                if v is not None and np.isfinite(v)]
        vmin, vmax = min([0.0] + vals), max([0.0] + vals)
        amp = (vmax - vmin) or 1.0
        for c, l in zip(centros, linhas):
            for (pol, _), d in zip(series, desloc):
                v = l.get(f"{pol}_{chave}")
                if v is None or not np.isfinite(v):
                    continue
                yi, v = c + d, v * escala
                if pol == "oraculo":
                    ax.barh(yi, v, height=alt, color="none", edgecolor=t["referencia"], linewidth=1.1,
                            linestyle=(0, (4, 3)))
                else:
                    ax.barh(yi, v, height=alt, color=t["destaque"] if pol == "dqn" else t["barra"])
                fim = max(v, 0.0)
                lo, hi = l.get(f"{pol}_{chave}_lo"), l.get(f"{pol}_{chave}_hi")
                if lo is not None and hi is not None and np.isfinite(lo) and hi > lo:
                    lo, hi = lo * escala, hi * escala
                    ax.plot([lo, hi], [yi, yi], color=t["texto"], linewidth=1.3)
                    for x in (lo, hi):
                        ax.plot([x, x], [yi - alt * 0.28, yi + alt * 0.28], color=t["texto"], linewidth=1.3)
                    fim = max(fim, hi)
                ax.text(fim + amp * 0.02, yi, num(v, casas) + "%", va="center", fontsize=11,
                        color=t["texto"] if pol != "oraculo" else t["secundario"],
                        fontweight="bold" if pol == "dqn" else "normal")
        ax.set_xlim(vmin - (amp * 0.04 if vmin < 0 else 0), vmax + amp * 0.2)
        if vmin < 0:
            ax.axvline(0, color=t["eixo"], linewidth=0.9)
        if escala == 100 and casas == 0 and vmin >= 0:
            ax.set_xticks([x for x in (0, 20, 40, 60, 80, 100) if x <= vmax + amp * 0.2])
        ax.set_title(titulo, color=t["texto"], fontsize=13, loc="left", pad=12)

    eixo0 = np.atleast_1d(eixos)[0]
    eixo0.set_yticks(centros)
    eixo0.set_yticklabels([NOMES_MODOS[l["modo"]].replace(" (", "\n(") for l in linhas], fontsize=12, color=t["texto"])

    cores = {"dqn": dict(facecolor=t["destaque"]), "oraculo": dict(facecolor="none", edgecolor=t["referencia"],
                                                                 linestyle=(0, (4, 3)))}
    itens = [Patch(label=rot, **cores.get(pol, dict(facecolor=t["barra"]))) for pol, rot in series]
    leg = fig.legend(handles=itens, loc="lower left", bbox_to_anchor=(0.2, 0.035), ncol=len(itens), frameon=False,
                     fontsize=11, labelcolor=t["secundario"], handlelength=1.2, columnspacing=2.0)
    for h in leg.legend_handles:
        h.set_height(9)
    fig.text(0.2, 0.95, subtitulo, color=t["secundario"], fontsize=10.5)
    if legenda_traco:
        fig.text(0.2 + 0.006, 0.012, legenda_traco, color=t["secundario"], fontsize=9.5)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.86, bottom=0.18, wspace=0.12)
    fig.savefig(saida, facecolor=t["fundo"])
    plt.close(fig)

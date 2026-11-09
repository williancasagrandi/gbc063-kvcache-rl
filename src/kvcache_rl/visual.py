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

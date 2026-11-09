"""Gera os gráficos dos resultados para os slides.

    python scripts/graficos.py --fonte gpt2
    python scripts/graficos.py --fonte gpt2 --tema claro

Lê resultados/<fonte>/baselines.csv e resultados/<fonte>/dqn/seed*/
e salva em resultados/<fonte>/figuras/:
    curva_aprendizado.png   retorno no teste ao longo do treino x políticas de referência
    comparacao_final.png    atenção perdida e agulha mantida por política
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kvcache_rl.baselines import NOMES

TEMAS = {
    # escuro = mesmo fundo dos slides
    "escuro": dict(fundo="#12161D", texto="#F1F5F9", secundario="#9AA7B8", grade="#262D3A",
                   eixo="#5B6778", barra="#5B6778", destaque="#C98500", referencia="#9AA7B8"),
    "claro": dict(fundo="#FFFFFF", texto="#0B0B0B", secundario="#52514E", grade="#E9E8E4",
                  eixo="#8A8880", barra="#B4B2AA", destaque="#2A78D6", referencia="#52514E"),
}
ORDEM = ["aleatoria", "janela_deslizante", "streaming_llm", "h2o", "dqn", "oraculo"]
FONTES = {"gpt2": "GPT-2", "sintetico": "sintéticos"}


def num(v: float, casas: int = 1) -> str:
    """Número no formato brasileiro, com sinal de menos tipográfico."""
    return f"{v:.{casas}f}".replace(".", ",").replace("-", "\u2212")


def ler_csv(caminho: Path) -> list[dict]:
    with open(caminho) as f:
        return [{k: (v if k == "politica" else float(v)) for k, v in linha.items()} for linha in csv.DictReader(f)]


def carregar(fonte: str):
    base = Path(f"resultados/{fonte}")
    baselines = {l["politica"]: l for l in ler_csv(base / "baselines.csv")}
    pastas = sorted((base / "dqn").glob("seed*"))
    curvas = [ler_csv(p / "curva.csv") for p in pastas if (p / "curva.csv").exists()]
    finais = [json.loads((p / "final.json").read_text()) for p in pastas if (p / "final.json").exists()]
    if not curvas:
        raise SystemExit(f"nenhuma curva em {base / 'dqn'}; rode scripts/treinar_dqn.py antes")
    n_teste = len(list(Path(f"dados/{fonte}/teste").glob("*.npz"))) or None
    return baselines, curvas, finais, n_teste


def estilo(ax, t):
    ax.set_facecolor(t["fundo"])
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(t["eixo"])
    ax.tick_params(colors=t["secundario"], labelsize=11, length=0)
    ax.grid(axis="y", color=t["grade"], linewidth=0.8)
    ax.set_axisbelow(True)


def espalhar(ys: list[float], folga: float) -> list[float]:
    """Afasta rótulos verticalmente para não se sobreporem (mantém a ordem)."""
    ordem = np.argsort(ys)
    pos = np.array(ys, dtype=float)[ordem]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + folga)
    centro = np.mean(ys) - np.mean(pos)
    pos += min(0.0, centro)  # recentraliza só se subiu demais
    out = np.empty_like(pos)
    out[ordem] = pos
    return out.tolist()


def curva_aprendizado(baselines, curvas, n_teste, t, saida: Path, fonte: str):
    passos = np.array([l["passos"] for l in curvas[0]]) / 1000
    n = min(len(c) for c in curvas)
    passos = passos[:n]
    ret = np.array([[l["retorno"] for l in c[:n]] for c in curvas])
    media, lo, hi = ret.mean(0), ret.min(0), ret.max(0)

    fig, ax = plt.subplots(figsize=(12, 5.4), dpi=200)
    fig.patch.set_facecolor(t["fundo"])
    estilo(ax, t)

    # referências: linhas horizontais em cinza, oráculo tracejado
    refs = [k for k in ORDEM if k in baselines]
    for k in refs:
        ax.axhline(baselines[k]["retorno"], color=t["referencia"], linewidth=1.2,
                   linestyle=(0, (5, 4)) if k == "oraculo" else "-", alpha=0.9 if k == "oraculo" else 0.55)

    if len(curvas) > 1:
        ax.fill_between(passos, lo, hi, color=t["destaque"], alpha=0.22, linewidth=0)
    ax.plot(passos, media, color=t["destaque"], linewidth=2.4)

    todos = np.concatenate([lo, hi, [baselines[k]["retorno"] for k in refs]])
    ymin, ymax = todos.min(), todos.max()
    margem = 0.08 * (ymax - ymin)
    ax.set_ylim(ymin - margem, ymax + margem)
    ax.set_xlim(0, passos[-1])

    # rótulos diretos à direita, sem sobreposição
    rotulos = [(f"{NOMES[k]}  {num(baselines[k]['retorno'])}", baselines[k]["retorno"], t["secundario"], False)
               for k in refs]
    rotulos.append((f"DQN  {num(media[-1])}", media[-1], t["destaque"], True))
    ys = espalhar([r[1] for r in rotulos], 0.055 * (ymax - ymin + 2 * margem))
    for (txt, _, cor, negrito), y in zip(rotulos, ys):
        ax.text(passos[-1] * 1.015, y, txt, color=cor, fontsize=12, va="center",
                fontweight="bold" if negrito else "normal", clip_on=False)

    ax.set_xlabel("passos de treino (mil)", color=t["secundario"], fontsize=12)
    ax.set_ylabel("retorno médio no teste", color=t["secundario"], fontsize=12)
    faixa = f" · faixa: mín–máx de {len(curvas)} sementes" if len(curvas) > 1 else ""
    textos = f"{n_teste} textos de teste" if n_teste else "textos de teste"
    fig.text(0.06, 0.94, f"DQN ao longo do treino, avaliado sem exploração em {textos} {FONTES.get(fonte, fonte)}{faixa}",
             color=t["secundario"], fontsize=12)
    fig.subplots_adjust(left=0.08, right=0.80, top=0.89, bottom=0.12)
    fig.savefig(saida, facecolor=t["fundo"])
    plt.close(fig)


def comparacao_final(baselines, finais, t, saida: Path):
    linhas = []
    for k in ORDEM:
        if k == "dqn":
            if not finais:
                continue
            perd = np.array([f["atencao_perdida"] for f in finais]) * 100
            agul = np.array([f["agulha"] for f in finais]) * 100
            linhas.append((k, perd.mean(), perd.min(), perd.max(), agul.mean(), agul.min(), agul.max()))
        elif k in baselines:
            b = baselines[k]
            linhas.append((k, b["atencao_perdida"] * 100, None, None, b["agulha"] * 100, None, None))

    fig, eixos = plt.subplots(1, 2, figsize=(12, 5.0), dpi=200, sharey=True)
    fig.patch.set_facecolor(t["fundo"])
    y = np.arange(len(linhas))[::-1]

    paineis = [
        (eixos[0], 1, "Atenção perdida por token (%) — menor é melhor", 1),
        (eixos[1], 4, "Agulha mantida até o fim (%) — maior é melhor", 0),
    ]
    for ax, col, titulo, casas in paineis:
        estilo(ax, t)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=t["grade"], linewidth=0.8)
        vmax = max(l[col + 2] if l[col + 2] is not None else l[col] for l in linhas)
        for yi, l in zip(y, linhas):
            k, v, lo, hi = l[0], l[col], l[col + 1], l[col + 2]
            if k == "oraculo":  # limite superior: só contorno
                ax.barh(yi, v, height=0.62, color="none", edgecolor=t["referencia"], linewidth=1.2,
                        linestyle=(0, (4, 3)))
            else:
                ax.barh(yi, v, height=0.62, color=t["destaque"] if k == "dqn" else t["barra"])
            if lo is not None and hi > lo:
                ax.plot([lo, hi], [yi, yi], color=t["texto"], linewidth=1.4)
                ax.plot([lo, lo], [yi - 0.12, yi + 0.12], color=t["texto"], linewidth=1.4)
                ax.plot([hi, hi], [yi - 0.12, yi + 0.12], color=t["texto"], linewidth=1.4)
            fim = hi if hi is not None else v
            ax.text(fim + vmax * 0.02, yi, num(v, casas) + "%", va="center", fontsize=12,
                    color=t["texto"], fontweight="bold" if k == "dqn" else "normal")
        ax.set_xlim(0, vmax * 1.22 if vmax > 0 else 1)
        if casas == 0:  # painel em %, sem marcas acima de 100
            ax.set_xticks([x for x in (0, 20, 40, 60, 80, 100) if x <= vmax * 1.22])
        ax.set_title(titulo, color=t["texto"], fontsize=13, loc="left", pad=12)

    eixos[0].set_yticks(y)
    eixos[0].set_yticklabels([NOMES[l[0]] for l in linhas], fontsize=12)
    for rot, l in zip(eixos[0].get_yticklabels(), linhas):
        rot.set_color(t["destaque"] if l[0] == "dqn" else t["texto"])
        rot.set_fontweight("bold" if l[0] == "dqn" else "normal")
    nota = "DQN: média de {} sementes, traço = mín–máx · Oráculo (tracejado) vê o futuro: limite superior".format(len(finais))
    fig.text(0.16, 0.03, nota, color=t["secundario"], fontsize=11)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.88, bottom=0.12, wspace=0.12)
    fig.savefig(saida, facecolor=t["fundo"])
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fonte", default="gpt2")
    p.add_argument("--tema", choices=list(TEMAS), default="escuro")
    a = p.parse_args()

    t = TEMAS[a.tema]
    from matplotlib import font_manager
    instaladas = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = next((f for f in ("Calibri", "Carlito") if f in instaladas), "DejaVu Sans")
    baselines, curvas, finais, n_teste = carregar(a.fonte)
    pasta = Path(f"resultados/{a.fonte}/figuras")
    pasta.mkdir(parents=True, exist_ok=True)
    sufixo = "" if a.tema == "escuro" else "_claro"
    curva_aprendizado(baselines, curvas, n_teste, t, pasta / f"curva_aprendizado{sufixo}.png", a.fonte)
    comparacao_final(baselines, finais, t, pasta / f"comparacao_final{sufixo}.png")
    print(f"figuras em {pasta}/")


if __name__ == "__main__":
    main()

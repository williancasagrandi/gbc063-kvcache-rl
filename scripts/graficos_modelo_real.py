"""Gráficos das métricas medidas no próprio LLM (depois de avaliar_no_modelo.py).

    python scripts/graficos_modelo_real.py --fonte qwen
    python scripts/graficos_modelo_real.py --fonte gpt2 --tema claro

Lê resultados/<fonte>/<modo>/modelo_real_textos.csv dos modos que tiverem e salva
em resultados/<fonte>/:
    comparacao_modos_real.png   aumento da perplexidade e senha recuperada, por modo,
                                para H2O, TOVA e DQN (traço = IC de 95% entre textos)
    proxy_x_real.png            atenção perdida (proxy) x KL do modelo completo (real),
                                um ponto por texto e política: o proxy prevê o real?
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from kvcache_rl.baselines import NOMES
from kvcache_rl.config import CHAVES, NOMES_MODOS
from kvcache_rl.visual import FONTES, TEMAS, estilo, grafico_modos, num, preparar_matplotlib

plt = preparar_matplotlib()
SERIES = [("h2o", "H2O"), ("tova", "TOVA"), ("dqn", "DQN")]
TRIVIAIS = ("aleatoria", "janela_deslizante")


def ler_textos(caminho: Path) -> list[dict]:
    with open(caminho) as f:
        return [{k: (v if k in ("texto", "politica") else float(v)) for k, v in l.items()} for l in csv.DictReader(f)]


def por_texto(linhas: list[dict], pol: str, chave: str) -> dict[str, float]:
    """Valor de cada texto para a política; o DQN é a média das sementes em cada texto."""
    sel = [l for l in linhas if (l["politica"].startswith("dqn_") if pol == "dqn" else l["politica"] == pol)]
    if chave == "acerto":  # a senha só conta onde o modelo acerta com o cache completo
        sel = [l for l in sel if l["acerto_cheio"] == 1.0]
    acc: dict[str, list[float]] = {}
    for l in sel:
        acc.setdefault(l["texto"], []).append(l[chave])
    return {t: float(np.mean(v)) for t, v in acc.items()}


def espalhar(ys: list[float], folga: float) -> list[float]:
    """Afasta valores na vertical para que fiquem a pelo menos `folga` um do outro (mantém a ordem)."""
    ordem = np.argsort(ys)
    pos = np.array(ys, dtype=float)[ordem]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + folga)
    pos -= (pos.mean() - np.mean(ys)) if len(pos) else 0  # recentraliza o grupo
    out = np.empty_like(pos)
    out[ordem] = pos
    return out.tolist()


def media_ic(v) -> tuple[float, float]:
    v = np.asarray(list(v), dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def coletar(fonte: str):
    modos = {}
    for modo in CHAVES:
        caminho = Path(f"resultados/{fonte}/{modo}/modelo_real_textos.csv")
        if caminho.exists():
            resumo = Path(f"resultados/{fonte}/{modo}/modelo_real_resumo.json")
            modos[modo] = (ler_textos(caminho), json.loads(resumo.read_text()) if resumo.exists() else {})
    if not modos:
        raise SystemExit(f"nenhum modelo_real_textos.csv em resultados/{fonte}/: rode scripts/avaliar_no_modelo.py")
    return modos


def linha_do_modo(modo: str, linhas: list[dict]) -> dict:
    l = {"modo": modo}
    for pol, _ in SERIES:
        m, ic = media_ic(por_texto(linhas, pol, "delta_nll").values())
        if np.isfinite(m):  # perplexidade: exp da NLL média; o IC vem do IC da ΔNLL
            l[f"{pol}_aumento_perplexidade"] = np.exp(m) - 1
            l[f"{pol}_aumento_perplexidade_lo"], l[f"{pol}_aumento_perplexidade_hi"] = np.exp(m - ic) - 1, np.exp(m + ic) - 1
        m, ic = media_ic(por_texto(linhas, pol, "acerto").values())
        if np.isfinite(m):
            l[f"{pol}_acerto"], l[f"{pol}_acerto_lo"], l[f"{pol}_acerto_hi"] = m, max(0.0, m - ic), min(1.0, m + ic)
    return l


def grafico_proxy_x_real(modos: dict, t: dict, saida: Path, fonte: str):
    fig, eixos = plt.subplots(1, len(modos), figsize=(12, 4.8), dpi=200, squeeze=False)
    fig.patch.set_facecolor(t["fundo"])
    for ax, (modo, (linhas, resumo)) in zip(eixos[0], modos.items()):
        estilo(ax, t)
        ax.grid(axis="x", color=t["grade"], linewidth=0.8)
        # só as competitivas: aleatória e janela deslizante (KL de 2 a 4 no Qwen) achatariam o resto
        comp = [l for l in linhas if l["politica"] not in TRIVIAIS]
        ax.scatter([l["atencao_perdida"] * 100 for l in comp], [l["kl"] for l in comp],
                   s=9, color=t["barra"], alpha=0.55, linewidths=0)
        pols = sorted({l["politica"] for l in comp if not l["politica"].startswith("dqn_")})
        pols += ["dqn"] if any(l["politica"].startswith("dqn_") for l in linhas) else []
        pontos = []
        for pol in pols:
            x = np.mean(list(por_texto(linhas, pol, "atencao_perdida").values())) * 100
            y = np.mean(list(por_texto(linhas, pol, "kl").values()))
            cor = t["destaque"] if pol == "dqn" else t["texto"]
            ax.scatter([x], [y], s=46, color=cor, edgecolors=t["fundo"], linewidths=1.0, zorder=3)
            pontos.append((pol, x, y, cor))
        # rótulos à direita dos pontos, afastados na vertical para não se sobreporem
        y0, y1 = ax.get_ylim()
        x0, x1 = ax.get_xlim()
        ys = espalhar([p[2] for p in pontos], 0.075 * (y1 - y0))
        for (pol, x, y, cor), yr in zip(pontos, ys):
            xr = x + 0.03 * (x1 - x0)
            if abs(yr - y) > 1e-9:
                ax.plot([x, xr], [y, yr], color=t["eixo"], linewidth=0.7, zorder=2)
            ax.text(xr, yr, NOMES.get(pol, pol), fontsize=9.5, va="center", color=cor,
                    fontweight="bold" if pol == "dqn" else "normal")
        ax.set_xlim(x0, x1 + 0.22 * (x1 - x0))  # espaço para os rótulos
        corr = resumo.get("correlacao_proxy", {}).get("kl", {})
        rho = corr.get("spearman_dentro_do_texto", float("nan")) if "politicas" in corr else float("nan")
        rho_txt = num(rho, 2) if np.isfinite(rho) else "— (rode avaliar_no_modelo.py)"
        ax.set_title(f"{NOMES_MODOS[modo]}\nSpearman dentro do texto: {rho_txt}", color=t["texto"],
                     fontsize=11.5, loc="left", pad=10)
        ax.set_xlabel("atenção perdida, proxy (%)", color=t["secundario"], fontsize=11)
    eixos[0][0].set_ylabel("KL do modelo completo, real (nats/token)", color=t["secundario"], fontsize=11)
    fig.text(0.02, 0.95, f"Cada ponto cinza é um texto de teste {FONTES.get(fonte, fonte)} sob uma política competitiva; "
             "os rotulados são médias · aleatória e janela deslizante ficam fora da escala",
             color=t["secundario"], fontsize=10.5)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.14, wspace=0.22)
    fig.savefig(saida, facecolor=t["fundo"])
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fonte", default="gpt2")
    p.add_argument("--tema", choices=list(TEMAS), default="escuro")
    a = p.parse_args()

    t, base = TEMAS[a.tema], Path(f"resultados/{a.fonte}")
    sufixo = "" if a.tema == "escuro" else "_claro"
    modos = coletar(a.fonte)
    linhas = [linha_do_modo(m, l) for m, (l, _) in modos.items()]

    paineis = [("aumento_perplexidade", "Aumento da perplexidade (%) — menor é melhor", 2, 100)]
    tem_senha = any(np.isfinite(l.get(f"{p}_acerto", np.nan)) for l in linhas for p, _ in SERIES)
    if tem_senha:
        paineis.append(("acerto", "Senha recuperada (%) — maior é melhor", 0, 100))
    textos = len({l["texto"] for l in next(iter(modos.values()))[0]})
    cheio = next(iter(modos.values()))[1].get("acerto_cache_completo", float("nan"))
    sub = f"Medido no próprio {FONTES.get(a.fonte, a.fonte)} com o cache podado · {textos} textos de teste"
    if tem_senha:
        sub += f" · senha: textos em que o modelo acerta com cache completo ({num(cheio * 100, 0)}%)"
    grafico_modos(plt, linhas, SERIES, paineis, t, base / f"comparacao_modos_real{sufixo}.png", sub,
                  "traço: intervalo de 95% entre textos (DQN: média das sementes em cada texto)")
    grafico_proxy_x_real(modos, t, base / f"proxy_x_real{sufixo}.png", a.fonte)

    print(f"{'modo':<28} {'política':<10} {'Δ perplexidade (IC 95%)':>28} {'senha recuperada':>18}")
    for l in linhas:
        for pol, _ in SERIES:
            if f"{pol}_aumento_perplexidade" not in l:
                continue
            ppl = (f"{l[f'{pol}_aumento_perplexidade'] * 100:6.2f}% ({l[f'{pol}_aumento_perplexidade_lo'] * 100:.2f} a "
                   f"{l[f'{pol}_aumento_perplexidade_hi'] * 100:.2f})")
            senha = f"{l[f'{pol}_acerto'] * 100:.0f}%" if f"{pol}_acerto" in l else "—"
            print(f"{NOMES_MODOS[l['modo']] if pol == 'h2o' else '':<28} {pol:<10} {ppl:>28} {senha:>18}")
    print(f"\nsalvo em {base}/comparacao_modos_real{sufixo}.png e proxy_x_real{sufixo}.png")


if __name__ == "__main__":
    main()

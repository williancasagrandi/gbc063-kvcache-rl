"""Compara os três modos (geral, por_camada, por_cabeca) nas métricas do PROXY.

As métricas no LLM real (perplexidade e senha) ficam em scripts/graficos_modelo_real.py.

    python scripts/comparar_modos.py --fonte gpt2

Para cada modo mostra o H2O (melhor regra fixa), o DQN e o oráculo, em
atenção perdida e agulha mantida. Usa os resultados já salvos por
avaliar_baselines.py e treinar_dqn.py; modos sem resultado são pulados.

Saída: resultados/<fonte>/comparacao_modos.png e comparacao_modos.csv
"""

import argparse
import csv
from pathlib import Path

from kvcache_rl.config import CHAVES, NOMES_MODOS
from kvcache_rl.visual import FONTES, TEMAS, carregar_resultados, grafico_modos, preparar_matplotlib, resumo_dqn

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
    for l in linhas:  # traço do DQN = mín–máx das sementes
        for chave in ("atencao_perdida", "agulha"):
            l[f"dqn_{chave}_lo"], l[f"dqn_{chave}_hi"] = l[f"dqn_{chave}_min"], l[f"dqn_{chave}_max"]
    grafico_modos(
        plt, linhas,
        series=[("h2o", "H2O (melhor regra fixa)"), ("dqn", "DQN"), ("oraculo", "Oráculo (vê o futuro)")],
        paineis=[("atencao_perdida", "Atenção perdida por token (%) — menor é melhor", 1, 100),
                 ("agulha", "Agulha no cache, proxy (%) — maior é melhor", 0, 100)],
        t=t, saida=saida,
        subtitulo=f"Métricas do ambiente proxy · mesmo orçamento (16 blocos) em cada cache · "
                  f"textos de teste {FONTES.get(fonte, fonte)}",
        legenda_traco="traço no DQN: mín–máx das sementes · métricas no LLM real: graficos_modelo_real.py")


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

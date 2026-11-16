"""Avalia as políticas no próprio LLM, com o cache podado (as métricas de verdade).

    python scripts/avaliar_no_modelo.py --fonte qwen --modo geral
    python scripts/avaliar_no_modelo.py --fonte qwen --modo por_cabeca
    python scripts/avaliar_no_modelo.py --fonte gpt2 --modo por_camada --n-textos 10   # rápido

Para cada texto de teste e cada política (5 regras fixas e o DQN de cada semente):
  1. roda a política no ambiente, em todas as unidades (camadas / cabeças de K/V),
     e anota quem foi despejado e quando;
  2. roda o LLM uma vez com a máscara correspondente, que equivale a apagar
     esses blocos do KV-cache (ver src/kvcache_rl/modelo_real.py);
  3. compara com o LLM de cache completo.

Métricas, por texto (médias nas previsões feitas depois que o cache enche):
  kl                  KL(p_cheio || p_podado) por token, em nats: quanto a distribuição
                      do próximo token mudou. É a recompensa densa proposta no Desafio 1.
  delta_nll           aumento da NLL por token; exp(delta_nll) - 1 = aumento da perplexidade.
  acerto              a decodificação gulosa ainda produz a senha inteira
                      (só conta nos textos em que o modelo acerta com cache completo).
  delta_nll_resposta  aumento da NLL dos tokens da senha (contínuo).
  atencao_perdida     a métrica do proxy, para medir se ela prevê as de cima.

Saída em resultados/<fonte>/<modo>/:
  modelo_real.csv          uma linha por política (e por semente do DQN; "dqn" = média)
  modelo_real_textos.csv   uma linha por texto e política
  modelo_real_resumo.json  correlação proxy x real e diferenças pareadas contra H2O e TOVA
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from kvcache_rl import KVCacheEnv, carregar_pasta
from kvcache_rl.baselines import NOMES as NOME
from kvcache_rl.baselines import politicas_ajustadas
from kvcache_rl.config import CHAVES, carregar_config
from kvcache_rl.llm import MODELOS, carregar_modelo, passada
from kvcache_rl.modelo_real import Mascara, despejos_do_texto, metricas, spearman, visibilidade_do_texto

POLITICAS = ["aleatoria", "janela_deslizante", "streaming_llm", "h2o", "tova"]
REFERENCIAS = ("h2o", "tova")  # as regras fortes contra as quais cada política é comparada texto a texto
CAMPOS = ["atencao_perdida", "kl", "delta_nll", "acerto", "delta_nll_resposta"]


def carregar_dqns(fonte: str, modo: str) -> dict:
    """Rede Q de cada semente treinada (resultados/<fonte>/<modo>/dqn/seed*/modelo.zip)."""
    from stable_baselines3 import DQN

    redes = {}
    for zip_ in sorted(Path(f"resultados/{fonte}/{modo}/dqn").glob("seed*/modelo.zip")):
        redes[f"dqn_{zip_.parent.name}"] = DQN.load(zip_, device="cpu").policy.q_net
    return redes


def media_ic(v) -> tuple[float, float]:
    """Média e meia-largura do intervalo de 95% (aproximação normal)."""
    v = np.asarray(list(v), dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def resumir(linhas: list[dict], nome: str) -> dict:
    sel = [l for l in linhas if l["politica"] == nome]
    validos = [l for l in sel if l["acerto_cheio"] == 1.0]  # a senha só é medida onde o modelo cheio acerta
    r = {"politica": nome, "textos": len(sel), "textos_senha": len(validos)}
    for c in CAMPOS:
        base = validos if c in ("acerto", "delta_nll_resposta") else sel
        r[c], r[f"{c}_ic95"] = media_ic(l[c] for l in base)
    r["aumento_perplexidade"] = float(np.exp(r["delta_nll"]) - 1)
    return r


# A aleatória e a janela deslizante (que apaga o bloco inicial) são tão piores que qualquer
# métrica as ordena certo; incluí-las inflaria a correlação. O que importa para treinar é se o
# proxy distingue as políticas competitivas entre si.
TRIVIAIS = ("aleatoria", "janela_deslizante")


def correlacao_dentro_do_texto(linhas: list[dict], chave: str, sem_triviais: bool = True) -> tuple[float, int]:
    """Spearman entre atenção perdida e `chave` entre as políticas de cada texto, média nos textos.
    Padrão: só as competitivas (StreamingLLM, H2O e cada semente do DQN). É o que importa para
    usar o proxy no treino: ele ordena essas políticas como o modelo real?"""
    rs = []
    for txt in dict.fromkeys(l["texto"] for l in linhas):
        sel = [l for l in linhas if l["texto"] == txt and not (sem_triviais and l["politica"] in TRIVIAIS)]
        rs.append(spearman([l["atencao_perdida"] for l in sel], [l[chave] for l in sel]))
    rs = [r for r in rs if np.isfinite(r)]
    return (float(np.mean(rs)) if rs else float("nan")), len(rs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fonte", default="gpt2")
    p.add_argument("--modo", default="geral", choices=list(CHAVES))
    p.add_argument("--config", default="configs/dqn.yml")
    p.add_argument("--modelo", default=None, help="sobrescreve o modelo gravado nos traces")
    p.add_argument("--politicas", nargs="+", default=POLITICAS, choices=POLITICAS)
    p.add_argument("--sem-dqn", action="store_true")
    p.add_argument("--n-textos", type=int, default=None, help="só os primeiros N textos de teste")
    p.add_argument("--dispositivo", default=None)
    p.add_argument("--precisao", default="float32", choices=["float32", "bfloat16", "float16", "auto"],
                   help="float32 (padrão) para as diferenças pequenas não se perderem no arredondamento")
    a = p.parse_args()

    cfg = carregar_config(a.config, a.modo)
    teste = carregar_pasta(f"dados/{a.fonte}/teste")[: a.n_textos]
    if any(tr.ids is None for tr in teste):
        raise SystemExit("os traces não têm os tokens do texto: gere-os de novo com scripts/gerar_traces_llm.py")
    env = KVCacheEnv(teste, **cfg["env"])

    nome_modelo = a.modelo or teste[0].modelo or MODELOS.get(a.fonte, {}).get("modelo")
    _, modelo, disp, arq = carregar_modelo(nome_modelo, a.dispositivo, a.precisao)
    forma = teste[0].cabecas.shape[:2]
    if forma != (arq.camadas, arq.kv):
        raise SystemExit(f"os traces têm {forma[0]}x{forma[1]} cabeças de K/V e {nome_modelo} tem "
                         f"{arq.camadas}x{arq.kv}: são de outro modelo")
    print(f"{nome_modelo} em {disp} ({a.precisao}): {arq}")

    # regras fixas com os parâmetros ajustados nos textos de treino (avaliar_baselines.py)
    ajustadas, achou = politicas_ajustadas(f"resultados/{a.fonte}/{a.modo}/ajuste.json")
    if achou:
        params = {n: p.keywords for n, p in ajustadas.items() if n in a.politicas and hasattr(p, "keywords")}
        print("regras fixas ajustadas no treino: " + ", ".join(f"{n} {p}" for n, p in params.items()))
    else:
        print("aviso: sem ajuste.json; regras fixas com parâmetros padrão "
              "(rode scripts/avaliar_baselines.py antes, para a comparação ser justa)")
    politicas = {n: dict(politica=ajustadas[n]) for n in a.politicas}
    if not a.sem_dqn:
        for nome, rede in carregar_dqns(a.fonte, a.modo).items():
            politicas[nome] = dict(q_net=rede)
    print(f"modo {a.modo}: {env.U} unidade(s), {len(teste)} textos, políticas: {', '.join(politicas)}")

    linhas, nb, bloco = [], env.traces[0].n_blocos, env.traces[0].bloco
    for i, tr in enumerate(tqdm(teste, desc="textos")):
        ids = list(tr.ids) + list(tr.resposta if tr.resposta is not None else [])
        n_resp = len(ids) - len(tr.ids)
        cheio = passada(modelo, ids, disp)
        acerto_cheio = metricas(cheio["nll"], cheio["certo"], cheio["nll"], nb, bloco, env.B, n_resp)["acerto"]
        for nome, pol in politicas.items():
            despejos, perdas = despejos_do_texto(env, i, **pol)
            with Mascara(modelo, visibilidade_do_texto(env, despejos), bloco, len(ids), arq):
                r = passada(modelo, ids, disp, referencia=cheio["h"])
            linhas.append({"texto": tr.nome, "politica": nome, "acerto_cheio": acerto_cheio,
                           "atencao_perdida": float(perdas.mean()),
                           **metricas(r["nll"], r["certo"], cheio["nll"], nb, bloco, env.B, n_resp, r["kl"])})

    # ---------------------------------------------------------------- resumo
    resumo = [resumir(linhas, n) for n in politicas]
    sementes = [r for r in resumo if r["politica"].startswith("dqn_")]
    if sementes:  # "dqn" = média das sementes, com mínimo e máximo
        dqn = {"politica": "dqn", "textos": sementes[0]["textos"], "textos_senha": sementes[0]["textos_senha"]}
        for c in CAMPOS + ["aumento_perplexidade"]:
            v = [s[c] for s in sementes]
            dqn[c], dqn[f"{c}_min"], dqn[f"{c}_max"] = float(np.mean(v)), float(np.min(v)), float(np.max(v))
        resumo.append(dqn)

    acerto_cheio = float(np.mean([l["acerto_cheio"] for l in linhas if l["politica"] == next(iter(politicas))]))
    extra = {"acerto_cache_completo": acerto_cheio, "correlacao_proxy": {}, "pareado": {}}
    for chave in ("kl", "delta_nll"):
        rho, n = correlacao_dentro_do_texto(linhas, chave)
        extra["correlacao_proxy"][chave] = {
            "spearman_dentro_do_texto": rho, "textos": n,
            "politicas": sorted({l["politica"] for l in linhas if l["politica"] not in TRIVIAIS}),
            "spearman_com_as_triviais": correlacao_dentro_do_texto(linhas, chave, sem_triviais=False)[0],
            "spearman_todos_os_pares": spearman([l["atencao_perdida"] for l in linhas], [l[chave] for l in linhas])}
    for ref in REFERENCIAS:  # mesmos textos: diferença texto a texto, com intervalo de 95%
        if ref not in politicas:
            continue
        base = {l["texto"]: l for l in linhas if l["politica"] == ref}
        extra["pareado"][ref] = {}
        for nome in politicas:
            if nome == ref:
                continue
            extra["pareado"][ref][nome] = {}
            for chave in ("kl", "delta_nll"):
                m, ic = media_ic(l[chave] - base[l["texto"]][chave] for l in linhas if l["politica"] == nome)
                extra["pareado"][ref][nome][chave] = {
                    "diferenca": m, "ic95": ic, "melhor": bool(m + ic < 0), "pior": bool(m - ic > 0)}

    base = Path(f"resultados/{a.fonte}/{a.modo}")
    base.mkdir(parents=True, exist_ok=True)
    with open(base / "modelo_real_textos.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    campos = sorted({k for r in resumo for k in r}, key=lambda k: (k != "politica", k))
    with open(base / "modelo_real.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(resumo)
    (base / "modelo_real_resumo.json").write_text(json.dumps(extra, indent=2))

    # ---------------------------------------------------------------- tabela
    print(f"\ncom o cache completo, o modelo acerta a senha em {acerto_cheio:.0%} dos textos de teste\n")
    print(f"{'política':<18} {'atenção perdida':>15} {'KL (nats/token)':>24} {'Δ perplexidade':>15} {'senha':>7}")
    for r in resumo:
        ic = f"± {r['kl_ic95']:.4f}" if "kl_ic95" in r else "(média sementes)"
        senha = "—" if not np.isfinite(r["acerto"]) else f"{r['acerto'] * 100:.0f}%"
        print(f"{r['politica']:<18} {r['atencao_perdida'] * 100:>14.2f}% {r['kl']:>8.4f} {ic:<15} "
              f"{r['aumento_perplexidade'] * 100:>14.2f}% {senha:>7}")
    if not np.isfinite(resumo[0]["acerto"]):
        print("(senha: —, o modelo não acerta a senha em nenhum texto nem com o cache completo)")

    c = extra["correlacao_proxy"]
    print(f"\no proxy ordena as políticas competitivas como o modelo real? Spearman dentro de cada texto "
          f"entre {', '.join(c['kl']['politicas'])} (média de {c['kl']['textos']} textos; 1 = mesma ordem):")
    print(f"  atenção perdida x KL: {c['kl']['spearman_dentro_do_texto']:.2f}    "
          f"atenção perdida x ΔNLL: {c['delta_nll']['spearman_dentro_do_texto']:.2f}    "
          f"(incluindo aleatória e janela deslizante, inflado: {c['kl']['spearman_com_as_triviais']:.2f})")
    for ref, comp in extra["pareado"].items():
        print(f"\ncontra o {NOME[ref]}, nos mesmos textos (KL da política − KL do {NOME[ref]}, IC de 95%):")
        for nome, d in comp.items():
            if nome in TRIVIAIS:
                continue
            d = d["kl"]
            veredito = "melhor" if d["melhor"] else "pior" if d["pior"] else "empate"
            print(f"  {nome:<18} {d['diferenca']:+.4f} ± {d['ic95']:.4f}  ({veredito})")
    print(f"\nsalvo em {base}/modelo_real.csv, modelo_real_textos.csv e modelo_real_resumo.json")


if __name__ == "__main__":
    main()

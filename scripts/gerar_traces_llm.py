"""Grava traces de atenção de um LLM real para treino e teste.

    python scripts/gerar_traces_llm.py --fonte gpt2             # GPT-2 small (MHA, 144 caches)
    python scripts/gerar_traces_llm.py --fonte qwen             # Qwen2.5-0.5B (GQA, 48 caches)
    python scripts/gerar_traces_llm.py --fonte qwen --tokens 2048 --exigir-acerto
    python scripts/gerar_traces_llm.py --fonte qwen --textos minha_pasta   # seus .txt

Saída: dados/<fonte>/treino/*.npz e dados/<fonte>/teste/*.npz.
Na primeira execução baixa o modelo (GPT-2 ~500 MB, Qwen2.5-0.5B ~1 GB) e a
WikiText-2 (~5 MB). Usa a GPU se houver; em CPU leva alguns minutos.

Com o cache completo o modelo deveria acertar a senha. O script mede essa taxa
e, com --exigir-acerto, guarda só os textos em que ele acerta.
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from kvcache_rl.llm import (MODELOS, PERGUNTA, PREFIXO_AGULHA, carregar_modelo, gerar_trace, montar_sequencia,
                            responde_certo)


def blocos_de_texto(tok, textos: list[str], tamanho: int, quantos: int) -> list[list[int]]:
    """Tokeniza os textos e corta em pedaços de `tamanho` tokens."""
    pedacos, buffer = [], []
    for t in textos:
        buffer.extend(tok(t).input_ids)
        while len(buffer) >= tamanho:
            pedacos.append(buffer[:tamanho])
            buffer = buffer[tamanho:]
            if len(pedacos) == quantos:
                return pedacos
    return pedacos


def ler_textos(args) -> list[str]:
    if args.textos:
        arquivos = sorted(Path(args.textos).glob("*.txt"))
        if not arquivos:
            raise SystemExit(f"nenhum .txt em {args.textos}")
        return [a.read_text(encoding="utf-8") for a in arquivos]
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    # junta as linhas em artigos (cada artigo começa com " = Título = ")
    artigos, atual = [], []
    for linha in ds["text"]:
        if linha.startswith(" = ") and not linha.startswith(" = = ") and atual:
            artigos.append("".join(atual))
            atual = []
        atual.append(linha)
    if atual:
        artigos.append("".join(atual))
    return artigos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fonte", default="gpt2", help=f"atalho de modelo e pasta: {', '.join(MODELOS)} (ou use --modelo)")
    p.add_argument("--modelo", default=None, help="nome no Hugging Face (sobrescreve o atalho)")
    p.add_argument("--textos", default=None, help="pasta com .txt (padrão: WikiText-2)")
    p.add_argument("--n-treino", type=int, default=200)
    p.add_argument("--n-teste", type=int, default=50)
    p.add_argument("--tokens", type=int, default=None, help="tamanho de cada texto (padrão: 1024)")
    p.add_argument("--bloco", type=int, default=16)
    p.add_argument("--saida", default=None, help="padrão: dados/<fonte>")
    p.add_argument("--dispositivo", default=None, help="cuda ou cpu (padrão: cuda se houver)")
    p.add_argument("--sem-verificar", action="store_true", help="não checar se o modelo acerta a senha")
    p.add_argument("--exigir-acerto", action="store_true", help="guardar só os textos em que o modelo acerta")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    atalho = MODELOS.get(a.fonte, {})
    nome_modelo = a.modelo or atalho.get("modelo")
    if not nome_modelo:
        raise SystemExit(f"fonte desconhecida: {a.fonte} (use {', '.join(MODELOS)} ou passe --modelo)")
    tokens = a.tokens or atalho.get("tokens", 1024)
    saida = Path(a.saida or f"dados/{a.fonte}")
    verificar = not a.sem_verificar or a.exigir_acerto

    rng = np.random.default_rng(a.seed)
    tok, modelo, disp, arq = carregar_modelo(nome_modelo, a.dispositivo)
    print(f"{nome_modelo} em {disp}: {arq}")
    if arq.max_tokens and tokens > arq.max_tokens:
        raise SystemExit(f"{nome_modelo} aceita no máximo {arq.max_tokens} tokens")
    nb = tokens // a.bloco
    mb = arq.unidades * nb * nb * 2 / 1e6
    print(f"{tokens} tokens = {nb} blocos de {a.bloco}; cada trace tem {arq.unidades} x {nb} x {nb} "
          f"valores por cabeça de K/V (~{mb:.1f} MB antes da compressão)")

    prefixo = tok(PREFIXO_AGULHA).input_ids
    ponto = tok(".").input_ids
    pergunta = tok(PERGUNTA).input_ids
    n_corpo = tokens - len(prefixo) - 3 - len(ponto) - len(pergunta)

    alvo = {"treino": a.n_treino, "teste": a.n_teste}
    total = a.n_treino + a.n_teste
    # com --exigir-acerto alguns textos são descartados, então pedimos folga de corpo
    corpos = blocos_de_texto(tok, ler_textos(a), n_corpo + 8, total * (4 if a.exigir_acerto else 1))
    if len(corpos) < total:
        raise SystemExit(f"texto insuficiente: {len(corpos)} pedaços para {total} traces")

    feitos = {"treino": 0, "teste": 0}
    acertos = tentativas = 0
    barra = tqdm(total=total, desc="traces")
    for corpo in corpos:
        parte = "treino" if feitos["treino"] < alvo["treino"] else "teste"
        if feitos[parte] >= alvo[parte]:
            break
        codigo_txt = str(int(rng.integers(1000, 10000)))
        codigo = tok(" " + codigo_txt).input_ids
        seq = montar_sequencia(corpo, prefixo, codigo, ponto, pergunta, tokens, a.bloco, rng, codigo_txt)

        if verificar:  # antes de gravar a atenção, que é a parte cara
            ok, _ = responde_certo(modelo, tok, seq, disp)
            tentativas, acertos = tentativas + 1, acertos + int(ok)
            barra.set_postfix(acerto=f"{acertos / tentativas:.0%}")
            if a.exigir_acerto and not ok:
                continue

        nome = f"{parte}_{feitos[parte]:04d}"
        gerar_trace(modelo, seq, a.bloco, disp, nome, arq.por_grupo).salvar(saida / parte / f"{nome}.npz")
        feitos[parte] += 1
        barra.update(1)
    barra.close()

    if feitos["teste"] < alvo["teste"]:
        taxa = f" (o modelo acertou a senha em {acertos}/{tentativas})" if tentativas else ""
        raise SystemExit(f"só deu para {feitos['treino']} traces de treino e {feitos['teste']} de teste{taxa}; "
                         "rode com mais textos ou sem --exigir-acerto")
    print(f"{feitos['treino']} traces de treino e {feitos['teste']} de teste em {saida}/")
    if tentativas:
        print(f"com o cache completo, o modelo acertou a senha em {acertos}/{tentativas} textos "
              f"({acertos / tentativas:.0%})")


if __name__ == "__main__":
    main()

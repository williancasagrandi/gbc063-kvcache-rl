"""Grava traces de atenção do GPT-2 para treino e teste.

    python scripts/gerar_traces_gpt2.py                      # textos da WikiText-2
    python scripts/gerar_traces_gpt2.py --textos minha_pasta # seus .txt

Saída: dados/gpt2/treino/*.npz e dados/gpt2/teste/*.npz.
Na primeira execução baixa o GPT-2 (~500 MB) e a WikiText-2 (~5 MB).
Usa a GPU se houver; em CPU leva alguns minutos.
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from kvcache_rl.gpt2 import PERGUNTA, PREFIXO_AGULHA, carregar_modelo, gerar_trace, montar_sequencia


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
    p.add_argument("--modelo", default="gpt2")
    p.add_argument("--textos", default=None, help="pasta com .txt (padrão: WikiText-2)")
    p.add_argument("--n-treino", type=int, default=200)
    p.add_argument("--n-teste", type=int, default=50)
    p.add_argument("--tokens", type=int, default=1024, help="tamanho de cada texto (máx. 1024 no GPT-2)")
    p.add_argument("--bloco", type=int, default=16)
    p.add_argument("--saida", default="dados/gpt2")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    tok, modelo, disp = carregar_modelo(a.modelo)
    print(f"modelo {a.modelo} em {disp}")

    prefixo = tok(PREFIXO_AGULHA).input_ids
    ponto = tok(".").input_ids
    pergunta = tok(PERGUNTA).input_ids
    n_corpo = a.tokens - len(prefixo) - 3 - len(ponto) - len(pergunta)

    total = a.n_treino + a.n_teste
    corpos = blocos_de_texto(tok, ler_textos(a), n_corpo + 8, total)
    if len(corpos) < total:
        raise SystemExit(f"texto insuficiente: {len(corpos)} pedaços para {total} traces")

    for i, corpo in enumerate(tqdm(corpos, desc="traces")):
        codigo_txt = str(int(rng.integers(1000, 10000)))
        codigo = tok(" " + codigo_txt).input_ids
        seq = montar_sequencia(corpo, prefixo, codigo, ponto, pergunta, a.tokens, a.bloco, rng, codigo_txt)
        parte = "treino" if i < a.n_treino else "teste"
        nome = f"{parte}_{i:04d}"
        tr = gerar_trace(modelo, seq, a.bloco, disp, nome)
        tr.salvar(Path(a.saida) / parte / f"{nome}.npz")

    print(f"{a.n_treino} traces de treino e {a.n_teste} de teste em {a.saida}/")


if __name__ == "__main__":
    main()

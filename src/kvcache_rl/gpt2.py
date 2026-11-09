"""Gravação de traces de atenção de um LLM real (padrão: GPT-2).

Cada texto recebe uma "agulha" (uma senha) numa posição aleatória do começo
e uma pergunta sobre ela no final. O modelo lê o texto uma única vez, com o
cache completo, e a atenção de cada cabeça de cada camada é agregada por bloco
(bloco-consulta x bloco-chave) e salva como um Trace, junto com as médias por
camada e do modelo todo.

As funções que dependem de torch/transformers importam essas bibliotecas
só quando chamadas, para o resto do pacote funcionar sem elas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .traces import Trace

# GPT-2 foi treinado em inglês, então a agulha e a pergunta também são em inglês.
PREFIXO_AGULHA = " Remember this: the secret code is"
PERGUNTA = " Question: what is the secret code? Answer: the secret code is"


@dataclass
class Sequencia:
    ids: list[int]
    bloco_agulha: int
    codigo: str


def montar_sequencia(
    corpo: list[int],
    prefixo: list[int],
    codigo: list[int],
    ponto: list[int],
    pergunta: list[int],
    total: int,
    bloco: int,
    rng: np.random.Generator,
    codigo_txt: str = "",
) -> Sequencia:
    """Monta corpo + agulha (em posição aleatória) + pergunta com `total` tokens.

    A agulha nunca cai nos dois primeiros blocos (o bloco 0 é o sink, que
    qualquer política razoável já mantém) e fica no primeiro quarto do texto.
    """
    if total % bloco:
        raise ValueError("total precisa ser múltiplo do bloco")
    agulha = prefixo + codigo + ponto
    n_corpo = total - len(agulha) - len(pergunta)
    if n_corpo <= 0 or len(corpo) < n_corpo:
        raise ValueError("corpo curto demais para o total pedido")
    corpo = corpo[:n_corpo]

    minimo = 2 * bloco
    maximo = max(minimo + 1, total // 4)
    p = int(rng.integers(minimo, maximo))
    ids = corpo[:p] + agulha + corpo[p:] + pergunta
    assert len(ids) == total

    fim_codigo = p + len(prefixo) + len(codigo) - 1  # último token do código
    return Sequencia(ids=ids, bloco_agulha=fim_codigo // bloco, codigo=codigo_txt)


def carregar_modelo(nome: str, dispositivo: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(nome)
    tok.model_max_length = 10**9  # tokenizamos artigos inteiros e cortamos depois; evita o aviso de tamanho
    modelo = AutoModelForCausalLM.from_pretrained(nome, attn_implementation="eager")
    return tok, modelo.to(dispositivo).eval(), dispositivo


def atencao_por_cabeca(modelo, ids: list[int], bloco: int, dispositivo: str) -> np.ndarray:
    """Roda o modelo uma vez e devolve a atenção por cabeça, agregada por bloco: (L, H, nb, nb)."""
    import torch

    cabecas = []
    with torch.no_grad():
        x = torch.tensor([ids], device=dispositivo)
        out = modelo(x, output_attentions=True)
        for a in out.attentions:  # (1, H, T, T) por camada
            m = a[0].float()
            m = m / m.sum(dim=-1, keepdim=True)  # garante linhas somando 1
            H, T, _ = m.shape
            nb = T // bloco
            # soma sobre os tokens de cada bloco-chave, média sobre os tokens de cada bloco-consulta
            m = m.reshape(H, nb, bloco, nb, bloco).sum(dim=-1).mean(dim=2)
            cabecas.append(m.cpu().numpy())
    return np.stack(cabecas)


def gerar_trace(modelo, seq: Sequencia, bloco: int, dispositivo: str, nome: str) -> Trace:
    cab = atencao_por_cabeca(modelo, seq.ids, bloco, dispositivo)
    camadas = cab.mean(axis=1)
    return Trace(att=camadas.mean(axis=0).astype(np.float32), bloco=bloco, agulha=seq.bloco_agulha,
                 nome=nome, camadas=camadas.astype(np.float16), cabecas=cab.astype(np.float16))

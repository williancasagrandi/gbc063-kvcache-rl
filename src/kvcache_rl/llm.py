"""Gravação de traces de atenção de um LLM real (GPT-2 ou Qwen2.5-0.5B).

Cada texto recebe uma "agulha" (uma senha) numa posição aleatória do começo
e uma pergunta sobre ela no final. O modelo lê o texto uma única vez, com o
cache completo, e a atenção de cada cabeça de cada camada é agregada por bloco
(bloco-consulta x bloco-chave) e salva como um Trace, junto com as médias por
camada e do modelo todo.

Atenção agrupada por cabeça de K/V
----------------------------------
O cache guarda um K e um V por **cabeça de K/V**, e é essa a unidade que pode
ser despejada. No GPT-2 (MHA) cada cabeça de consulta tem o seu próprio K/V,
então as duas contagens coincidem: 12 camadas x 12 cabeças = 144 caches.
O Qwen2.5-0.5B usa GQA: 14 cabeças de consulta por camada compartilham 2 de
K/V (7 consultas por cabeça de K/V), então são 24 x 2 = 48 caches. Como um
token só pode sair do cache para o grupo inteiro, a atenção das cabeças de
consulta de um grupo é mediada: é o que aquele cache "serve" em média.

As funções que dependem de torch/transformers importam essas bibliotecas
só quando chamadas, para o resto do pacote funcionar sem elas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .traces import Trace

# Os modelos foram treinados sobretudo em inglês, então a agulha e a pergunta também são em inglês.
PREFIXO_AGULHA = " Remember this: the secret code is"
PERGUNTA = " Question: what is the secret code? Answer: the secret code is"

# Atalhos de linha de comando: --fonte gpt2 / --fonte qwen
MODELOS = {
    "gpt2": dict(modelo="gpt2", tokens=1024, descricao="GPT-2 small, atenção multi-cabeça (MHA)"),
    "qwen": dict(modelo="Qwen/Qwen2.5-0.5B", tokens=1024, descricao="Qwen2.5-0.5B, atenção agrupada (GQA)"),
}


@dataclass
class Sequencia:
    ids: list[int]
    bloco_agulha: int
    codigo: str


@dataclass
class Arquitetura:
    """O que precisamos saber do modelo para agrupar a atenção e dimensionar os traces."""

    camadas: int
    consultas: int  # cabeças de consulta (Q) por camada
    kv: int  # cabeças de K/V por camada = caches por camada
    max_tokens: int

    @property
    def por_grupo(self) -> int:
        """Quantas cabeças de consulta compartilham cada cache (1 = MHA, sem GQA)."""
        return self.consultas // self.kv

    @property
    def unidades(self) -> int:
        return self.camadas * self.kv

    def __str__(self) -> str:
        tipo = "MHA" if self.por_grupo == 1 else f"GQA, {self.por_grupo} consultas por cabeça de K/V"
        return (f"{self.camadas} camadas x {self.kv} cabeça(s) de K/V = {self.unidades} caches "
                f"({self.consultas} cabeças de consulta por camada; {tipo})")


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


def agrupar_kv(att: np.ndarray, por_grupo: int) -> np.ndarray:
    """(H, ...) por cabeça de consulta -> (H / por_grupo, ...) por cabeça de K/V.

    O HF repete cada cabeça de K/V `por_grupo` vezes em sequência (repeat_kv), então
    a cabeça de consulta `q` usa o cache `q // por_grupo`. Tiramos a média do grupo:
    as linhas continuam somando 1 e o valor é a atenção que aquele cache serve em média.
    """
    if por_grupo <= 1:
        return att
    H = att.shape[0]
    if H % por_grupo:
        raise ValueError(f"{H} cabeças não dividem em grupos de {por_grupo}")
    return att.reshape(H // por_grupo, por_grupo, *att.shape[1:]).mean(axis=1)


def ler_arquitetura(modelo) -> Arquitetura:
    import warnings

    c = modelo.config
    camadas = getattr(c, "num_hidden_layers", None) or c.n_layer
    consultas = getattr(c, "num_attention_heads", None) or c.n_head
    kv = getattr(c, "num_key_value_heads", None) or consultas  # MHA: cada consulta tem o seu K/V
    maximo = getattr(c, "max_position_embeddings", None) or getattr(c, "n_positions", 0)
    if getattr(c, "use_sliding_window", False):
        # com janela deslizante o modelo já descarta contexto sozinho, e o problema deixa de ser o nosso
        warnings.warn("este modelo usa janela deslizante: a atenção gravada não é global", RuntimeWarning)
    return Arquitetura(camadas=int(camadas), consultas=int(consultas), kv=int(kv), max_tokens=int(maximo))


def carregar_modelo(nome: str, dispositivo: str | None = None):
    """Devolve (tokenizador, modelo, dispositivo, arquitetura)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging

    logging.set_verbosity_error()  # silencia avisos repetidos de generate()
    dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
    # na GPU, bfloat16 (ou float16) cabe folgado e é mais rápido; na CPU, float32 é mais rápido
    if dispositivo == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tok = AutoTokenizer.from_pretrained(nome)
    tok.model_max_length = 10**9  # tokenizamos artigos inteiros e cortamos depois; evita o aviso de tamanho
    # eager: só essa implementação devolve a matriz de atenção que precisamos gravar
    modelo = AutoModelForCausalLM.from_pretrained(nome, attn_implementation="eager", torch_dtype=dtype)
    modelo = modelo.to(dispositivo).eval()
    return tok, modelo, dispositivo, ler_arquitetura(modelo)


class _Gravador:
    """Agrega a atenção de cada camada por bloco e libera a matriz grande na hora.

    Com `output_attentions=True`, o HF guarda uma matriz (cabeças, T, T) por camada
    até o fim do forward. No Qwen com 4.096 tokens isso daria 24 x 14 x 4.096² em
    float32, mais de 20 GB. O hook agrega a matriz de cada camada para
    (cabeças de K/V, nb, nb) e devolve None no lugar dela, que é liberada na hora:
    o pico passa a ser o de uma camada só, cabeça por cabeça.
    """

    def __init__(self, bloco: int, por_grupo: int):
        self.bloco, self.por_grupo = bloco, por_grupo
        self.camadas: list[np.ndarray] = []

    def _agregar(self, pesos) -> np.ndarray:
        import torch

        m = pesos[0] if pesos.dim() == 4 else pesos  # (cabeças, T, T)
        H, T, _ = m.shape
        nb = T // self.bloco
        saida = []
        for h in range(H):  # cabeça por cabeça para não materializar tudo em float32
            x = m[h].float()
            x = x / x.sum(dim=-1, keepdim=True)  # garante linhas somando 1
            # soma sobre os tokens de cada bloco-chave, média sobre os tokens de cada bloco-consulta
            saida.append(x.reshape(nb, self.bloco, nb, self.bloco).sum(dim=-1).mean(dim=1))
        att = torch.stack(saida).cpu().numpy()
        return agrupar_kv(att, self.por_grupo)

    def hook(self, modulo, entrada, saida):
        """Guarda a atenção desta camada e a tira da saída (versões do HF variam na aridade)."""
        import torch

        if not isinstance(saida, tuple):
            return None
        novo, achou = list(saida), False
        for i, t in enumerate(novo):
            if torch.is_tensor(t) and t.dim() == 4 and t.shape[-1] == t.shape[-2]:
                self.camadas.append(self._agregar(t))
                novo[i], achou = None, True
        return tuple(novo) if achou else None


def _modulos_de_atencao(modelo) -> list:
    """Os módulos de atenção, camada por camada (Qwen: model.layers[i].self_attn; GPT-2: transformer.h[i].attn)."""
    for caminho in ("model.layers", "transformer.h"):
        obj = modelo
        for parte in caminho.split("."):
            obj = getattr(obj, parte, None)
            if obj is None:
                break
        if obj is not None:
            return [getattr(c, "self_attn", None) or c.attn for c in obj]
    raise RuntimeError(f"não sei onde estão as camadas de {type(modelo).__name__}")


def atencao_por_cabeca(modelo, ids: list[int], bloco: int, dispositivo: str, por_grupo: int = 1) -> np.ndarray:
    """Roda o modelo uma vez e devolve a atenção agregada por bloco: (camadas, cabeças de K/V, nb, nb)."""
    import torch

    g = _Gravador(bloco, por_grupo)
    ganchos = [m.register_forward_hook(g.hook) for m in _modulos_de_atencao(modelo)]
    try:
        with torch.no_grad():
            saida = modelo(torch.tensor([ids], device=dispositivo), output_attentions=True, use_cache=False)
    finally:
        for h in ganchos:
            h.remove()

    if len(g.camadas) == len(ganchos):
        return np.stack(g.camadas)
    # fallback: se o hook não achou a matriz (versão inesperada do HF), usa a saída do modelo.
    # Funciona igual, mas guarda todas as camadas de uma vez: pode faltar memória em textos longos.
    if not saida.attentions or saida.attentions[0] is None:
        raise RuntimeError("não foi possível obter a matriz de atenção do modelo")
    import warnings

    warnings.warn("o hook não capturou a atenção; usando a saída do modelo (mais memória)", RuntimeWarning)
    return np.stack([g._agregar(a) for a in saida.attentions])


def gerar_trace(modelo, seq: Sequencia, bloco: int, dispositivo: str, nome: str, por_grupo: int = 1) -> Trace:
    cab = atencao_por_cabeca(modelo, seq.ids, bloco, dispositivo, por_grupo)
    camadas = cab.mean(axis=1)
    return Trace(att=camadas.mean(axis=0).astype(np.float32), bloco=bloco, agulha=seq.bloco_agulha,
                 nome=nome, camadas=camadas.astype(np.float16), cabecas=cab.astype(np.float16))


def responde_certo(modelo, tok, seq: Sequencia, dispositivo: str, max_novos: int = 8) -> tuple[bool, str]:
    """Com o cache COMPLETO, o modelo acerta a senha? Mede se a tarefa faz sentido para ele."""
    import torch

    with torch.no_grad():
        x = torch.tensor([seq.ids], device=dispositivo)
        saida = modelo.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=max_novos,
                                do_sample=False, pad_token_id=tok.eos_token_id)
    resposta = tok.decode(saida[0, len(seq.ids):], skip_special_tokens=True)
    return seq.codigo in resposta, resposta.strip()

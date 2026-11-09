"""Traces de atenção agregados por bloco.

Um trace guarda, para cada token-consulta t de um texto, quanta atenção ele deu
a cada bloco de tokens anteriores, calculada com o cache COMPLETO. O ambiente
usa isso para simular despejos sem rodar o LLM a cada passo.

Formato (.npz):
    att     -> float32 (T, nb): att[t, j] = soma da atenção do token t sobre os
               tokens do bloco j, média sobre camadas e cabeças. Linhas somam ~1
               e att[t, j] = 0 para j > t // bloco (causal).
    camadas -> float16 (L, T, nb): o mesmo, separado por camada (média sobre as
               cabeças). Usado pelos modos "grupos" e "por_camada".
    bloco   -> int, tokens por bloco
    agulha  -> int, índice do bloco com a "agulha" (-1 se não houver)
    nome    -> str
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Trace:
    att: np.ndarray
    bloco: int
    agulha: int = -1
    nome: str = ""
    camadas: np.ndarray | None = None  # (L, T, nb), float16

    @property
    def n_tokens(self) -> int:
        return self.att.shape[0]

    @property
    def n_blocos(self) -> int:
        return self.att.shape[1]

    @property
    def n_camadas(self) -> int:
        return 0 if self.camadas is None else self.camadas.shape[0]

    def salvar(self, caminho: str | Path) -> None:
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        extra = {} if self.camadas is None else {"camadas": self.camadas.astype(np.float16)}
        np.savez_compressed(
            caminho,
            att=self.att.astype(np.float32),
            bloco=np.int64(self.bloco),
            agulha=np.int64(self.agulha),
            nome=np.str_(self.nome),
            **extra,
        )

    @staticmethod
    def carregar(caminho: str | Path) -> "Trace":
        d = np.load(caminho, allow_pickle=False)
        return Trace(
            att=d["att"].astype(np.float32),
            bloco=int(d["bloco"]),
            agulha=int(d["agulha"]),
            nome=str(d["nome"]),
            camadas=d["camadas"] if "camadas" in d.files else None,
        )


def carregar_pasta(pasta: str | Path) -> list[Trace]:
    arquivos = sorted(Path(pasta).glob("*.npz"))
    if not arquivos:
        raise FileNotFoundError(f"nenhum trace .npz em {pasta}")
    return [Trace.carregar(a) for a in arquivos]


def agregar_por_bloco(att_tokens: np.ndarray, bloco: int) -> np.ndarray:
    """(T, T) atenção token->token  ->  (T, nb) atenção token->bloco."""
    T = att_tokens.shape[0]
    nb = int(np.ceil(T / bloco))
    pad = nb * bloco - T
    if pad:
        att_tokens = np.pad(att_tokens, ((0, 0), (0, pad)))
    return att_tokens.reshape(T, nb, bloco).sum(axis=2)


def agrupar_camadas(trace: Trace, grupos: int) -> np.ndarray:
    """(L, T, nb) -> (grupos, T, nb): média das camadas consecutivas de cada grupo."""
    if trace.camadas is None:
        raise ValueError(f"o trace {trace.nome!r} não tem atenção por camada: gere os traces de novo")
    L = trace.n_camadas
    if not 1 <= grupos <= L:
        raise ValueError(f"grupos deve estar entre 1 e {L}")
    partes = np.array_split(np.arange(L), grupos)
    return np.stack([trace.camadas[p].astype(np.float32).mean(axis=0) for p in partes])


def gerar_sintetico(
    n_blocos: int = 64,
    bloco: int = 16,
    rng: np.random.Generator | None = None,
    com_agulha: bool = True,
    nome: str = "sintetico",
    n_camadas: int = 12,
) -> Trace:
    """Gera um trace com os padrões que aparecem em LLMs reais, camada a camada.

    Em todas as camadas:
    - sink: o primeiro bloco recebe atenção o tempo todo;
    - recência: blocos próximos recebem mais atenção (decaimento exponencial);
    - heavy hitters comuns: alguns blocos importam durante todo o texto;
    - agulha: um bloco do início que fica quieto e só é consultado no final.

    Diferenças entre camadas (como nos LLMs, onde cada camada faz uma coisa):
    - camadas iniciais olham mais para perto; as finais, mais para o sink;
    - cada família de camadas vizinhas tem seus próprios heavy hitters;
    - revisitas (blocos que voltam a ser consultados) aparecem só em algumas camadas;
    - a consulta à agulha no final é forte nas camadas do meio e finais.
    """
    rng = rng or np.random.default_rng()
    nb, L = n_blocos, n_camadas
    T = nb * bloco
    idx = np.arange(nb)
    dist = idx[:, None] - idx[None, :]          # bloco-consulta x bloco-chave
    visivel = dist >= 0
    qb_token = np.arange(T) // bloco

    agulha = int(rng.integers(1, max(2, nb // 5))) if com_agulha else -1

    hh_comuns = rng.choice(np.arange(1, nb), size=max(1, nb // 16), replace=False)
    peso_comum = rng.uniform(0.6, 1.4, size=hh_comuns.size)
    n_familias = 4
    hh_familia = [rng.choice(np.arange(1, nb), size=max(1, nb // 16), replace=False) for _ in range(n_familias)]
    peso_familia = [rng.uniform(1.0, 2.5, size=h.size) for h in hh_familia]

    # revisitas: (bloco alvo, início, duração, peso, primeira camada, última camada + 1)
    revisitas = []
    for _ in range(nb // 6):
        alvo = int(rng.integers(1, nb - 4))
        inicio = int(rng.integers(alvo + 4, nb))
        c0 = int(rng.integers(0, L))
        revisitas.append((alvo, inicio, int(rng.integers(1, 4)), rng.uniform(1.5, 4.0), c0, c0 + int(rng.integers(3, 7))))

    camadas = np.zeros((L, T, nb), dtype=np.float64)
    for c in range(L):
        frac = c / max(1, L - 1)
        tau = 0.8 + 3.0 * frac + rng.uniform(-0.3, 0.3)
        base = np.zeros(nb)
        base[hh_comuns] += peso_comum
        familia = min(n_familias - 1, c * n_familias // L)
        base[hh_familia[familia]] += peso_familia[familia]
        if agulha >= 0:
            base[agulha] = 0.0  # fica quieta até a pergunta

        W = np.where(visivel, 3.0 * np.exp(-np.clip(dist, 0, None) / tau) + base[None, :], 0.0)
        W[:, 0] += 1.0 + 4.0 * frac
        for alvo, ini, dur, peso, c0, c1 in revisitas:
            if c0 <= c < c1:
                W[ini:ini + dur, alvo] += peso
        if agulha >= 0:
            W[nb - 2:, agulha] += 6.0 if frac >= 0.3 else 1.0

        Wt = W[qb_token] * rng.gamma(4.0, 0.25, size=(T, nb))  # ruído; zeros continuam zeros
        camadas[c] = Wt / Wt.sum(axis=1, keepdims=True)

    att = camadas.mean(axis=0)
    return Trace(att=att.astype(np.float32), bloco=bloco, agulha=agulha, nome=nome,
                 camadas=camadas.astype(np.float16))

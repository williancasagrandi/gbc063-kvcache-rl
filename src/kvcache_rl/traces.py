"""Traces de atenção agregados por bloco.

Um trace guarda, para cada token-consulta t de um texto, quanta atenção ele deu
a cada bloco de tokens anteriores, calculada com o cache COMPLETO. O ambiente
usa isso para simular despejos sem rodar o LLM a cada passo.

Formato (.npz):
    att    -> float32 (T, nb): att[t, j] = soma da atenção do token t sobre os
              tokens do bloco j (média sobre camadas e cabeças). Linhas somam ~1
              e att[t, j] = 0 para j > t // bloco (causal).
    bloco  -> int, tokens por bloco
    agulha -> int, índice do bloco com a "agulha" (-1 se não houver)
    nome   -> str
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

    @property
    def n_tokens(self) -> int:
        return self.att.shape[0]

    @property
    def n_blocos(self) -> int:
        return self.att.shape[1]

    def salvar(self, caminho: str | Path) -> None:
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            caminho,
            att=self.att.astype(np.float32),
            bloco=np.int64(self.bloco),
            agulha=np.int64(self.agulha),
            nome=np.str_(self.nome),
        )

    @staticmethod
    def carregar(caminho: str | Path) -> "Trace":
        d = np.load(caminho, allow_pickle=False)
        return Trace(
            att=d["att"].astype(np.float32),
            bloco=int(d["bloco"]),
            agulha=int(d["agulha"]),
            nome=str(d["nome"]),
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


def gerar_sintetico(
    n_blocos: int = 64,
    bloco: int = 16,
    rng: np.random.Generator | None = None,
    com_agulha: bool = True,
    nome: str = "sintetico",
) -> Trace:
    """Gera um trace com os padrões que aparecem em LLMs reais.

    - sink: o primeiro bloco recebe muita atenção o tempo todo;
    - recência: blocos próximos recebem mais atenção (decaimento exponencial);
    - heavy hitters: alguns blocos são importantes durante todo o texto;
    - revisitas: blocos "adormecidos" voltam a ser consultados mais tarde
      (é isso que quebra heurísticas que só olham o passado recente);
    - agulha: um bloco do início que só é consultado de novo no final.
    """
    rng = rng or np.random.default_rng()
    T = n_blocos * bloco

    base = np.zeros(n_blocos)
    hh = rng.choice(np.arange(1, n_blocos), size=max(1, n_blocos // 16), replace=False)
    base[hh] = rng.uniform(0.6, 1.4, size=hh.size)

    # revisitas: (bloco alvo, início, duração em blocos, peso)
    revisitas = []
    for _ in range(n_blocos // 6):
        alvo = int(rng.integers(1, n_blocos - 4))
        inicio = int(rng.integers(alvo + 4, n_blocos))
        revisitas.append((alvo, inicio, int(rng.integers(1, 4)), rng.uniform(1.5, 4.0)))

    agulha = -1
    if com_agulha:
        agulha = int(rng.integers(1, max(2, n_blocos // 5)))
        base[agulha] = 0.0  # fica quieta até a pergunta

    tau = rng.uniform(1.5, 3.5)
    att = np.zeros((T, n_blocos), dtype=np.float64)
    idx = np.arange(n_blocos)
    for t in range(T):
        qb = t // bloco
        w = np.zeros(n_blocos)
        vis = idx <= qb
        w[vis] = 3.0 * np.exp(-(qb - idx[vis]) / tau) + base[vis]
        w[0] += 2.5
        for alvo, ini, dur, peso in revisitas:
            if ini <= qb < ini + dur:
                w[alvo] += peso
        if agulha >= 0 and qb >= n_blocos - 2:
            w[agulha] += 6.0
        w[vis] *= rng.gamma(4.0, 0.25, size=vis.sum())
        w[~vis] = 0.0
        att[t] = w / w.sum()

    return Trace(att=att.astype(np.float32), bloco=bloco, agulha=agulha, nome=nome)

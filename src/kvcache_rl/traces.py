"""Traces de atenção agregados por bloco.

Um trace guarda quanta atenção cada bloco de tokens (as consultas) deu a cada
bloco anterior (as chaves), calculada com o cache COMPLETO. O ambiente usa isso
para simular despejos sem rodar o LLM a cada passo.

Como o agente só decide nas fronteiras de bloco, basta guardar a atenção média
dos tokens de cada bloco, em vez de token a token: a perda, a atenção acumulada,
a recente e a futura saem exatamente iguais, e o arquivo fica 16x menor.

Formato (.npz), com nb = número de blocos:
    att     -> float32 (nb, nb): att[k, j] = atenção média que os tokens do bloco
               k dão aos tokens do bloco j, média sobre camadas e cabeças.
               Linhas somam 1 e att[k, j] = 0 para j > k (causal).
    camadas -> float16 (L, nb, nb): o mesmo por camada (média das cabeças).
    cabecas -> float16 (L, H, nb, nb): o mesmo por cabeça de atenção.
    bloco   -> int, tokens por bloco
    agulha  -> int, bloco com a "agulha" (-1 se não houver)
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
    camadas: np.ndarray | None = None  # (L, nb, nb), float16
    cabecas: np.ndarray | None = None  # (L, H, nb, nb), float16

    @property
    def n_blocos(self) -> int:
        return self.att.shape[0]

    @property
    def n_tokens(self) -> int:
        return self.n_blocos * self.bloco

    def salvar(self, caminho: str | Path) -> None:
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        extra = {}
        if self.camadas is not None:
            extra["camadas"] = self.camadas.astype(np.float16)
        if self.cabecas is not None:
            extra["cabecas"] = self.cabecas.astype(np.float16)
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
        att = d["att"].astype(np.float32)
        if att.shape[0] != att.shape[1]:
            raise ValueError(f"{caminho}: formato antigo (atenção por token); gere os traces de novo")
        return Trace(
            att=att,
            bloco=int(d["bloco"]),
            agulha=int(d["agulha"]),
            nome=str(d["nome"]),
            camadas=d["camadas"] if "camadas" in d.files else None,
            cabecas=d["cabecas"] if "cabecas" in d.files else None,
        )


def carregar_pasta(pasta: str | Path) -> list[Trace]:
    arquivos = sorted(Path(pasta).glob("*.npz"))
    if not arquivos:
        raise FileNotFoundError(f"nenhum trace .npz em {pasta}")
    return [Trace.carregar(a) for a in arquivos]


def agregar_blocos(att: np.ndarray, bloco: int) -> np.ndarray:
    """(..., T, T) atenção token->token  ->  (..., nb, nb) atenção bloco->bloco.

    Soma a atenção sobre os tokens de cada bloco-chave e tira a média sobre os
    tokens de cada bloco-consulta. T precisa ser múltiplo de `bloco`.
    """
    T = att.shape[-1]
    if T % bloco:
        raise ValueError("o número de tokens precisa ser múltiplo do bloco")
    nb = T // bloco
    x = att.reshape(*att.shape[:-2], nb, bloco, nb, bloco)
    return x.sum(axis=-1).mean(axis=-2)


def perfil_cabecas(n_camadas: int, n_cabecas: int, semente: int = 2026) -> dict:
    """Papel fixo de cada cabeça no modelo sintético (como os pesos de um modelo treinado):
    é o mesmo em todos os textos, assim como no GPT-2 a mesma cabeça faz o mesmo papel."""
    rng = np.random.default_rng(semente)
    L, H = n_camadas, n_cabecas
    frac = np.repeat(np.arange(L)[:, None] / max(1, L - 1), H, axis=1)
    tipo = rng.choice(["local", "sink", "hh", "revisita"], size=(L, H), p=[0.35, 0.2, 0.3, 0.15])
    tau = np.where(tipo == "local", rng.uniform(0.6, 1.2, (L, H)), rng.uniform(1.5, 4.0, (L, H)))
    sink = 1.0 + 4.0 * frac + np.where(tipo == "sink", rng.uniform(3.0, 8.0, (L, H)), 0.0)
    familia = rng.integers(0, 4, (L, H))
    # poucas cabeças de "recuperação", nas camadas do meio para o fim: são elas que buscam a agulha
    candidatas = [(l, h) for l in range(L // 3, L) for h in range(H)]
    n_rec = max(1, round(0.04 * L * H))
    escolhidas = rng.choice(len(candidatas), size=min(n_rec, len(candidatas)), replace=False)
    recuperacao = np.zeros((L, H), dtype=bool)
    for e in escolhidas:
        recuperacao[candidatas[e]] = True
    return dict(tipo=tipo, tau=tau, sink=sink, familia=familia, recuperacao=recuperacao)


def gerar_sintetico(
    n_blocos: int = 64,
    bloco: int = 16,
    rng: np.random.Generator | None = None,
    com_agulha: bool = True,
    nome: str = "sintetico",
    n_camadas: int = 12,
    n_cabecas: int = 12,
) -> Trace:
    """Gera um trace com padrões de LLMs reais, cabeça a cabeça.

    Cada cabeça tem um papel fixo (ver perfil_cabecas):
    - local: olha quase só para os blocos mais próximos;
    - sink: despeja muita atenção no primeiro bloco;
    - hh: consulta sempre os mesmos blocos importantes (heavy hitters) do texto;
    - revisita: volta a consultar blocos que ficaram quietos por um tempo.
    Em todas: recência e sink que cresce com a profundidade da camada.
    A agulha fica num bloco do começo, quieta até o final, quando só as poucas
    cabeças de recuperação a consultam com força.
    """
    rng = rng or np.random.default_rng()
    nb, L, H = n_blocos, n_camadas, n_cabecas
    p = perfil_cabecas(L, H)
    idx = np.arange(nb)
    dist = idx[:, None] - idx[None, :]  # bloco-consulta x bloco-chave
    visivel = dist >= 0

    agulha = int(rng.integers(2, max(3, nb // 4))) if com_agulha else -1
    hh_comuns = rng.choice(np.arange(1, nb), size=max(1, nb // 16), replace=False)
    peso_comum = rng.uniform(0.6, 1.4, size=hh_comuns.size)
    hh_familia = [rng.choice(np.arange(1, nb), size=max(1, nb // 16), replace=False) for _ in range(4)]
    peso_familia = [rng.uniform(1.0, 2.5, size=h.size) for h in hh_familia]
    revisitas = []
    for _ in range(nb // 6):
        alvo = int(rng.integers(1, nb - 4))
        inicio = int(rng.integers(alvo + 4, nb))
        revisitas.append((alvo, inicio, int(rng.integers(1, 4)), rng.uniform(1.5, 4.0)))

    cab = np.zeros((L, H, nb, nb), dtype=np.float64)
    for l in range(L):
        for h in range(H):
            tipo = p["tipo"][l, h]
            base = np.zeros(nb)
            base[hh_comuns] += peso_comum
            if tipo == "hh":
                f = p["familia"][l, h]
                base[hh_familia[f]] += 2.0 * peso_familia[f]
            if agulha >= 0:
                base[agulha] = 0.0  # quieta até a pergunta
            W = np.where(visivel, 3.0 * np.exp(-np.clip(dist, 0, None) / p["tau"][l, h]) + base[None, :], 0.0)
            W[:, 0] += p["sink"][l, h]
            if tipo == "revisita":
                for alvo, ini, dur, peso in revisitas:
                    W[ini:ini + dur, alvo] += peso
            if agulha >= 0:
                W[nb - 2:, agulha] += 8.0 if p["recuperacao"][l, h] else 0.2
            W = W * rng.gamma(16.0, 1 / 16, size=W.shape)  # ruído; zeros continuam zeros
            cab[l, h] = W / W.sum(axis=1, keepdims=True)

    camadas = cab.mean(axis=1)
    return Trace(att=camadas.mean(axis=0).astype(np.float32), bloco=bloco, agulha=agulha, nome=nome,
                 camadas=camadas.astype(np.float16), cabecas=cab.astype(np.float16))

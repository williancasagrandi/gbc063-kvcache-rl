"""Ambiente Gymnasium de despejo de KV-cache (versão proxy).

Dinâmica de um episódio (um texto):
  1. Os primeiros `orcamento` blocos entram no cache sem nenhuma decisão.
  2. Cada novo bloco é processado (seus tokens consultam o cache + ele mesmo).
     Ao final do bloco, o cache tem orcamento + 1 blocos e o agente escolhe
     um para despejar. Despejado, o bloco nunca mais volta.
  3. A recompensa do passo é menos a massa média de atenção que os tokens do
     próximo bloco dariam aos blocos já despejados (atenção "perdida").
  4. No fim do texto, se a agulha ainda está no cache, soma-se `bonus_agulha`.

Ação: índice em {0, ..., orcamento}. Os candidatos ficam ordenados por posição
no texto, então o índice `orcamento` é sempre o bloco que acabou de chegar.

Observação: para cada candidato, 5 features (ver FEATURES) + progresso global.
"""

from __future__ import annotations

from typing import Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .traces import Trace

FEATURES = ("idade", "atencao_acumulada", "atencao_recente", "eh_sink", "posicao")


class KVCacheEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        traces: Sequence[Trace],
        orcamento: int = 16,
        janela_recente: int = 32,
        n_sink: int = 1,
        bonus_agulha: float = 10.0,
        render_mode: str | None = None,
    ):
        super().__init__()
        if not traces:
            raise ValueError("passe ao menos um trace")
        for tr in traces:
            if tr.n_blocos <= orcamento + 1:
                raise ValueError(f"trace {tr.nome!r} tem poucos blocos para orcamento={orcamento}")
        self.traces = list(traces)
        self.B = orcamento
        self.janela = janela_recente
        self.n_sink = n_sink
        self.bonus_agulha = bonus_agulha
        self.render_mode = render_mode

        self.n_cand = self.B + 1
        self.n_feat = len(FEATURES)
        dim = self.n_cand * self.n_feat + 1
        # features de atenção são escaladas por n_cand (atenção uniforme = 1), então ficam em [0, n_cand]
        self.observation_space = spaces.Box(low=0.0, high=float(self.n_cand), shape=(dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_cand)

        self.trace: Trace | None = None

    # ------------------------------------------------------------------ API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        if "trace" in options:
            self.trace = self.traces[int(options["trace"])]
        else:
            self.trace = self.traces[int(self.np_random.integers(len(self.traces)))]

        tr = self.trace
        self.bs = tr.bloco
        self.nb = tr.n_blocos
        self.T = tr.n_tokens

        self.cache: list[int] = []
        self.despejados = np.zeros(self.nb, dtype=bool)
        self.acum = np.zeros(self.nb, dtype=np.float64)
        self.n_vistos = np.zeros(self.nb, dtype=np.int64)
        self.t = 0
        self.atencao_perdida = 0.0

        # blocos 0..B entram e são processados sem decisão
        for k in range(self.B + 1):
            self.cache.append(k)
            self._processar_bloco(k)
        self.k = self.B  # último bloco que entrou

        return self._obs(), self._info()

    def step(self, acao):
        acao = int(acao)
        if not self.action_space.contains(acao):
            raise ValueError(f"ação inválida: {acao}")

        bloco_saiu = self.cache.pop(acao)
        self.despejados[bloco_saiu] = True

        self.k += 1
        self.cache.append(self.k)
        perdida = self._processar_bloco(self.k)
        self.atencao_perdida += perdida
        recompensa = -perdida

        terminou = self.k == self.nb - 1
        agulha_ok = None
        if terminou and self.trace.agulha >= 0:
            agulha_ok = self.trace.agulha in self.cache
            if agulha_ok:
                recompensa += self.bonus_agulha

        info = self._info()
        info["bloco_despejado"] = bloco_saiu
        if terminou:
            n_dec = self.nb - 1 - self.B
            info["atencao_perdida_media"] = self.atencao_perdida / n_dec
            info["agulha_mantida"] = agulha_ok
        return self._obs(), float(recompensa), terminou, False, info

    def render(self):
        if self.render_mode != "ansi":
            return None
        marca = []
        for j in range(self.nb):
            if j in self.cache:
                marca.append("A" if j == self.trace.agulha else "#")
            elif self.despejados[j]:
                marca.append(".")
            else:
                marca.append(" ")
        return f"[{''.join(marca)}] bloco {self.k}/{self.nb - 1}"

    # ------------------------------------------------------- auxiliares
    def _processar_bloco(self, k: int) -> float:
        """Roda os tokens do bloco k. Retorna a atenção média perdida por token."""
        ini, fim = k * self.bs, min((k + 1) * self.bs, self.T)
        linhas = self.trace.att[ini:fim]
        perdida = float(linhas[:, self.despejados].sum(axis=1).mean()) if self.despejados.any() else 0.0
        ids = np.array(self.cache)
        self.acum[ids] += linhas[:, ids].sum(axis=0)
        self.n_vistos[ids] += fim - ini
        self.t = fim
        return perdida

    def _features(self) -> np.ndarray:
        ids = np.array(self.cache)
        escala = float(self.n_cand)  # atenção uniforme vira ~1
        idade = (self.k - ids) / self.nb
        acum = np.where(self.n_vistos[ids] > 0, self.acum[ids] / np.maximum(self.n_vistos[ids], 1), 0.0) * escala
        ini = max(0, self.t - self.janela)
        recente = self.trace.att[ini:self.t, ids].mean(axis=0) * escala
        sink = (ids < self.n_sink).astype(np.float64)
        pos = ids / self.nb
        return np.stack([idade, acum, recente, sink, pos], axis=1)

    def _obs(self) -> np.ndarray:
        f = self._features().reshape(-1)
        return np.concatenate([f, [self.t / self.T]]).astype(np.float32)

    def _info(self) -> dict:
        f = self._features()
        return {
            "candidatos": np.array(self.cache),
            "idade": f[:, 0],
            "atencao_acumulada": f[:, 1],
            "atencao_recente": f[:, 2],
        }

    # usado só pelo oráculo (olha o futuro, serve de limite superior)
    def massa_futura(self) -> np.ndarray:
        ids = np.array(self.cache)
        return self.trace.att[self.t:, ids].sum(axis=0)

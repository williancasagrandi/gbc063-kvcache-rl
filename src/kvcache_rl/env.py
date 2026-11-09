"""Ambiente Gymnasium de despejo de KV-cache (versão proxy).

Dinâmica de um episódio:
  1. Os primeiros `orcamento` blocos entram no cache sem nenhuma decisão.
  2. Cada novo bloco é processado (seus tokens consultam o cache + ele mesmo).
     Ao final do bloco, o cache tem orcamento + 1 blocos e o agente escolhe
     um para despejar. Despejado, o bloco nunca mais volta.
  3. A recompensa é menos a massa média de atenção que os tokens do próximo
     bloco dariam aos blocos já despejados (atenção "perdida").
  4. No fim do texto, soma-se `bonus_agulha` se a agulha ainda está no cache.

Três modos, sempre com o mesmo orçamento de memória por camada:

  geral       um cache para todas as camadas; o estado usa a atenção média
              de todas as camadas. Episódio = um texto.
  grupos      um cache para todas as camadas, mas o estado separa a atenção
              por grupos de camadas consecutivas (ex.: iniciais, meio, finais).
              Episódio = um texto.
  por_camada  cada grupo de camadas tem o seu próprio cache (com grupos = nº de
              camadas, um cache por camada). Episódio = um texto visto por UMA
              camada, com a perda e a agulha daquela camada; o estado diz qual
              camada é. O mesmo agente serve para todas as camadas (parâmetros
              compartilhados). Avaliar um texto = rodar todas as camadas e
              tirar a média, então os números ficam na mesma escala dos outros
              modos (agulha = fração das camadas que a mantiveram).

Cada camada é um episódio separado porque os caches das camadas não interagem:
juntar as decisões de várias camadas no mesmo episódio faria o "próximo estado"
de uma decisão ser o cache de outra camada, que o agente não vê.

Ação: índice em {0, ..., orcamento}. Os candidatos ficam ordenados por posição
no texto, então o índice `orcamento` é sempre o bloco que acabou de chegar.

Observação, por candidato: idade, sink, posição e, para cada fonte de atenção
(1 nos modos geral e por_camada, G no modo grupos), atenção acumulada e recente.
Depois vem o progresso no texto e, no modo por_camada, qual camada é (one-hot).
"""

from __future__ import annotations

from typing import Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .traces import Trace, agrupar_camadas

MODOS = ("geral", "grupos", "por_camada")


class KVCacheEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        traces: Sequence[Trace],
        orcamento: int = 16,
        janela_recente: int = 32,
        n_sink: int = 1,
        bonus_agulha: float = 10.0,
        modo: str = "geral",
        grupos: int = 3,
        render_mode: str | None = None,
    ):
        super().__init__()
        if not traces:
            raise ValueError("passe ao menos um trace")
        if modo not in MODOS:
            raise ValueError(f"modo deve ser um de {MODOS}")
        for tr in traces:
            if tr.n_blocos <= orcamento + 1:
                raise ValueError(f"trace {tr.nome!r} tem poucos blocos para orcamento={orcamento}")
        self.traces = list(traces)
        self.B = orcamento
        self.janela = janela_recente
        self.n_sink = n_sink
        self.bonus_agulha = bonus_agulha
        self.modo = modo
        self.render_mode = render_mode

        # atenção usada pelo ambiente: (G, T, nb) por trace
        if modo == "geral":
            self.G = 1
            self._att = [tr.att[None] for tr in self.traces]
        else:
            self.G = grupos
            self._att = [agrupar_camadas(tr, grupos) for tr in self.traces]

        self.n_fontes = self.G if modo == "grupos" else 1
        self.n_cand = self.B + 1
        self.n_feat = 3 + 2 * self.n_fontes
        dim = self.n_cand * self.n_feat + 1 + (self.G if modo == "por_camada" else 0)
        # features de atenção são escaladas por n_cand (atenção uniforme = 1), então ficam em [0, n_cand]
        self.observation_space = spaces.Box(low=0.0, high=float(self.n_cand), shape=(dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_cand)

        self.trace: Trace | None = None
        self.camada = 0

    @property
    def decisoes_por_texto(self) -> int:
        """Decisões para processar um texto inteiro (todas as camadas, no modo por_camada)."""
        por_cache = self.traces[0].n_blocos - 1 - self.B
        return por_cache * (self.G if self.modo == "por_camada" else 1)

    def episodios_avaliacao(self) -> list[dict]:
        """Opções de reset que cobrem cada trace uma vez (e cada camada, no modo por_camada)."""
        if self.modo == "por_camada":
            return [{"trace": i, "camada": c} for i in range(len(self.traces)) for c in range(self.G)]
        return [{"trace": i} for i in range(len(self.traces))]

    # ------------------------------------------------------------------ API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        i = int(options["trace"]) if "trace" in options else int(self.np_random.integers(len(self.traces)))
        self.trace = self.traces[i]
        if self.modo == "por_camada":
            c = int(options["camada"]) if "camada" in options else int(self.np_random.integers(self.G))
            self.camada = c
            self.A = self._att[i][c:c + 1]  # (1, T, nb): só a camada deste episódio
        else:
            self.camada = 0
            self.A = self._att[i]  # (G, T, nb)

        tr = self.trace
        self.bs, self.nb, self.T = tr.bloco, tr.n_blocos, tr.n_tokens
        self.cache: list[int] = []
        self.despejados = np.zeros(self.nb, dtype=bool)
        self.acum = np.zeros((self.A.shape[0], self.nb), dtype=np.float64)
        self.n_vistos = np.zeros(self.nb, dtype=np.int64)
        self.t = 0
        self.perda_total = 0.0

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
        perda = self._processar_bloco(self.k)
        self.perda_total += perda
        recompensa = -perda

        terminou = self.k == self.nb - 1
        agulha = None
        if terminou and self.trace.agulha >= 0:
            agulha = float(self.trace.agulha in self.cache)
            recompensa += self.bonus_agulha * agulha

        info = self._info()
        info["bloco_despejado"] = bloco_saiu
        if terminou:
            info["atencao_perdida_media"] = self.perda_total / (self.nb - 1 - self.B)
            info["agulha_mantida"] = agulha
        return self._obs(), float(recompensa), bool(terminou), False, info

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
        return f"[{''.join(marca)}] camada {self.camada} bloco {self.k}/{self.nb - 1}"

    # ------------------------------------------------------- auxiliares
    def _processar_bloco(self, k: int) -> float:
        """Roda os tokens do bloco k. Retorna a atenção média perdida por token
        (média também sobre os grupos de camadas, quando há mais de um)."""
        ini, fim = k * self.bs, min((k + 1) * self.bs, self.T)
        linhas = self.A[:, ini:fim]  # (fontes, tokens, nb)
        perda = float(linhas[:, :, self.despejados].sum(axis=2).mean()) if self.despejados.any() else 0.0
        ids = np.array(self.cache)
        self.acum[:, ids] += linhas[:, :, ids].sum(axis=1)
        self.n_vistos[ids] += fim - ini
        self.t = fim
        return perda

    def _features(self) -> np.ndarray:
        ids = np.array(self.cache)
        escala = float(self.n_cand)  # atenção uniforme vira ~1
        idade = (self.k - ids) / self.nb
        sink = (ids < self.n_sink).astype(np.float64)
        pos = ids / self.nb
        acum = self.acum[:, ids] / np.maximum(self.n_vistos[ids], 1) * escala  # (fontes, cand)
        ini = max(0, self.t - self.janela)
        recente = self.A[:, ini:self.t][:, :, ids].mean(axis=1) * escala
        return np.column_stack([idade, sink, pos, acum.T, recente.T])

    def _obs(self) -> np.ndarray:
        partes = [self._features().reshape(-1), [self.t / self.T]]
        if self.modo == "por_camada":
            partes.append(np.eye(self.G)[self.camada])
        return np.concatenate(partes).astype(np.float32)

    def _info(self) -> dict:
        f = self._features()
        nf = self.n_fontes
        return {
            "candidatos": np.array(self.cache),
            "camada": self.camada,
            "idade": f[:, 0],
            "atencao_acumulada": f[:, 3:3 + nf].mean(axis=1),
            "atencao_recente": f[:, 3 + nf:].mean(axis=1),
        }

    # usado só pelo oráculo (olha o futuro, serve de limite superior)
    def massa_futura(self) -> np.ndarray:
        ids = np.array(self.cache)
        return self.A[:, self.t:][:, :, ids].sum(axis=1).mean(axis=0)

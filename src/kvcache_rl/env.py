"""Ambiente Gymnasium de despejo de KV-cache (versão proxy).

Dinâmica de um episódio:
  1. Os primeiros `orcamento` blocos entram no cache sem nenhuma decisão.
  2. Cada novo bloco é processado (seus tokens consultam o cache + ele mesmo).
     Ao final do bloco, o cache tem orcamento + 1 blocos e o agente escolhe
     um para despejar. Despejado, o bloco nunca mais volta.
  3. A recompensa é menos a atenção média que os tokens do próximo bloco dariam
     aos blocos já despejados (atenção "perdida"). O retorno do episódio é, então,
     menos a atenção perdida acumulada: é o único objetivo do agente.
  4. (Opcional, desligado por padrão) bônus no fim do texto se a agulha ainda está
     no cache. Desligado porque usa informação que a política não teria (onde está
     a agulha) e dominava o retorno; a agulha é medida na avaliação, no LLM real
     (scripts/avaliar_no_modelo.py), e não entra no treino.

Três modos, sempre com o mesmo orçamento (16 blocos) em cada cache:

  geral       um cache para o modelo todo; o estado usa a atenção média de
              todas as camadas e cabeças. Episódio = um texto.
  por_camada  um cache por camada. Episódio = um texto visto por UMA camada,
              com a atenção daquela camada; o estado diz qual camada é.
  por_cabeca  um cache por cabeça de K/V, que é a unidade que o cache guarda de
              fato (GPT-2: 12 x 12 = 144; Qwen2.5-0.5B, com GQA: 24 x 2 = 48).
              Episódio = um texto visto por UMA cabeça; o estado diz qual é.

Nos modos por camada e por cabeça, o mesmo agente serve para todas as unidades
(parâmetros compartilhados). Cada unidade é um episódio separado porque os
caches não interagem: juntar várias unidades no mesmo episódio faria o "próximo
estado" de uma decisão ser o cache de outra unidade, que o agente não vê.

Diagnóstico da agulha no proxy (não entra na recompensa por padrão): a métrica
"agulha" de um texto é a fração da atenção que a pergunta final dá à agulha que
foi preservada (no modo geral é 0 ou 1; por camada/cabeça, a média das unidades
ponderada pelo quanto cada uma consulta a agulha). Se `bonus_agulha` > 0, o
bônus de cada unidade é `bonus_agulha` vezes essa relevância normalizada.

Ação: índice em {0, ..., orcamento}. Os candidatos ficam ordenados por posição
no texto, então o índice `orcamento` é sempre o bloco que acabou de chegar.

Observação, por candidato: idade, sink, posição, atenção acumulada e atenção
recente. Depois vem o progresso no texto e, nos modos por camada e por cabeça,
qual unidade é (one-hot).
"""

from __future__ import annotations

from typing import Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .traces import Trace

MODOS = ("geral", "por_camada", "por_cabeca")
FEATURES = ("idade", "eh_sink", "posicao", "atencao_acumulada", "atencao_recente")


class KVCacheEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        traces: Sequence[Trace],
        orcamento: int = 16,
        janela_recente: int = 2,
        n_sink: int = 1,
        bonus_agulha: float = 0.0,
        modo: str = "geral",
        render_mode: str | None = None,
    ):
        """janela_recente: em blocos (2 blocos = 32 tokens)."""
        super().__init__()
        if not traces:
            raise ValueError("passe ao menos um trace")
        if modo not in MODOS:
            raise ValueError(f"modo deve ser um de {MODOS}")
        for tr in traces:
            if tr.n_blocos <= orcamento + 1:
                raise ValueError(f"trace {tr.nome!r} tem poucos blocos para orcamento={orcamento}")
            if modo == "por_camada" and tr.camadas is None or modo == "por_cabeca" and tr.cabecas is None:
                raise ValueError(f"o trace {tr.nome!r} não tem a atenção que o modo {modo} precisa: gere os traces de novo")
        self.traces = list(traces)
        self.B = orcamento
        self.janela = janela_recente
        self.n_sink = n_sink
        self.bonus_agulha = bonus_agulha
        self.modo = modo
        self.render_mode = render_mode

        tr0 = self.traces[0]
        if modo == "geral":
            self.U, self.forma = 1, ()
        elif modo == "por_camada":
            self.U, self.forma = tr0.camadas.shape[0], (tr0.camadas.shape[0],)
        else:
            L, H = tr0.cabecas.shape[:2]
            self.U, self.forma = L * H, (L, H)

        # pesos da agulha por unidade: relevância (máx = 1, usada no bônus) e peso (soma = 1, usada na métrica)
        self._relevancia, self._peso = [], []
        for tr in self.traces:
            w = self._atencao_na_agulha(tr)
            self._relevancia.append(w / w.max() if w.max() > 0 else np.ones(self.U))
            self._peso.append(w / w.sum() if w.sum() > 0 else np.full(self.U, 1 / self.U))

        self.n_cand = self.B + 1
        dim = self.n_cand * len(FEATURES) + 1 + (self.U if modo != "geral" else 0)
        # features de atenção são escaladas por n_cand (atenção uniforme = 1), então ficam em [0, n_cand]
        self.observation_space = spaces.Box(low=0.0, high=float(self.n_cand), shape=(dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_cand)

        self.trace: Trace | None = None
        self.unidade = 0

    # --------------------------------------------------------- unidades
    def _matriz(self, i: int, u: int) -> np.ndarray:
        tr = self.traces[i]
        if self.modo == "geral":
            return tr.att
        if self.modo == "por_camada":
            return tr.camadas[u].astype(np.float32)
        return tr.cabecas[u // self.forma[1], u % self.forma[1]].astype(np.float32)

    def _atencao_na_agulha(self, tr: Trace) -> np.ndarray:
        """Atenção que o último bloco (a pergunta) dá à agulha, em cada unidade."""
        if tr.agulha < 0:
            return np.ones(self.U)
        if self.modo == "geral":
            return np.ones(1)
        if self.modo == "por_camada":
            return tr.camadas[:, -1, tr.agulha].astype(np.float64)
        return tr.cabecas[:, :, -1, tr.agulha].astype(np.float64).reshape(-1)

    def nome_unidade(self, u: int) -> str:
        if self.modo == "geral":
            return "modelo"
        if self.modo == "por_camada":
            return f"camada {u}"
        return f"camada {u // self.forma[1]}, cabeça de K/V {u % self.forma[1]}"

    @property
    def decisoes_por_texto(self) -> int:
        """Decisões para processar um texto inteiro em todas as unidades."""
        return (self.traces[0].n_blocos - 1 - self.B) * self.U

    def episodios_avaliacao(self) -> list[dict]:
        """Opções de reset que cobrem cada trace em cada unidade uma vez."""
        return [{"trace": i, "unidade": u} for i in range(len(self.traces)) for u in range(self.U)]

    # ------------------------------------------------------------------ API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        i = int(options["trace"]) if "trace" in options else int(self.np_random.integers(len(self.traces)))
        u = int(options["unidade"]) if "unidade" in options else int(self.np_random.integers(self.U))
        self.i, self.unidade = i, u
        self.trace = self.traces[i]
        self.A = self._matriz(i, u)  # (nb, nb)
        self.nb = self.trace.n_blocos

        self._onehot = np.zeros(self.U if self.modo != "geral" else 0, dtype=np.float32)
        if self.modo != "geral":
            self._onehot[u] = 1.0
        self.cache: list[int] = []
        self.despejados = np.zeros(self.nb, dtype=bool)
        self.acum = np.zeros(self.nb, dtype=np.float64)
        self.n_vistos = np.zeros(self.nb, dtype=np.int64)
        self.perda_total = 0.0

        # blocos 0..B entram e são processados sem decisão
        for k in range(self.B + 1):
            self.cache.append(k)
            self._processar_bloco(k)
        self.k = self.B  # último bloco que entrou

        f = self._features()
        return self._obs(f), self._info(f)

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
        f = self._features()
        info = self._info(f)
        info["bloco_despejado"] = bloco_saiu
        if terminou:
            info["atencao_perdida_media"] = self.perda_total / (self.nb - 1 - self.B)
            if self.trace.agulha >= 0:
                mantida = float(self.trace.agulha in self.cache)
                recompensa += self.bonus_agulha * mantida * float(self._relevancia[self.i][self.unidade])
                info["agulha_mantida"] = mantida
                info["peso_agulha"] = float(self._peso[self.i][self.unidade])
        return self._obs(f), float(recompensa), bool(terminou), False, info

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
        return f"[{''.join(marca)}] {self.nome_unidade(self.unidade)}, bloco {self.k}/{self.nb - 1}"

    # ------------------------------------------------------- auxiliares
    def _processar_bloco(self, k: int) -> float:
        """Os tokens do bloco k consultam o cache. Retorna a atenção média perdida por token."""
        linha = self.A[k]
        perda = float(linha[self.despejados].sum())
        ids = np.array(self.cache)
        self.acum[ids] += linha[ids]
        self.n_vistos[ids] += 1
        return perda

    def _features(self) -> np.ndarray:
        ids = np.array(self.cache)
        escala = float(self.n_cand)  # atenção uniforme vira ~1
        idade = (self.k - ids) / self.nb
        sink = (ids < self.n_sink).astype(np.float64)
        pos = ids / self.nb
        acum = self.acum[ids] / np.maximum(self.n_vistos[ids], 1) * escala
        recente = self.A[max(0, self.k - self.janela + 1):self.k + 1, ids].mean(axis=0) * escala
        return np.column_stack([idade, sink, pos, acum, recente])

    def _obs(self, f: np.ndarray) -> np.ndarray:
        return np.concatenate([f.reshape(-1), [(self.k + 1) / self.nb], self._onehot]).astype(np.float32)

    def _info(self, f: np.ndarray) -> dict:
        return {
            "candidatos": np.array(self.cache),
            "unidade": self.unidade,
            "idade": f[:, 0],
            "atencao_acumulada": f[:, 3],
            "atencao_recente": f[:, 4],
        }

    # usado só pelo oráculo (olha o futuro, serve de limite superior)
    def massa_futura(self) -> np.ndarray:
        ids = np.array(self.cache)
        return self.A[self.k + 1:, ids].sum(axis=0)

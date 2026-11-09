"""Avaliação das políticas no próprio LLM, com o cache podado.

A atenção perdida do ambiente proxy é só uma aproximação: ela soma a atenção que
os blocos apagados receberiam, mas não diz o que acontece com o texto que o modelo
prevê. Aqui medimos isso diretamente.

Despejar um bloco do KV-cache equivale a proibir que as consultas seguintes o
vejam. Então os despejos de uma política viram uma máscara de atenção, e um único
forward com essa máscara dá exatamente o que o modelo calcularia se os blocos
tivessem sido apagados do cache durante a leitura: o K/V de cada token é calculado
uma vez, a partir do que aquele token via quando foi lido, e as posições dos que
ficam não mudam (como no H2O e no SnapKV). Ao contrário do proxy, aqui o softmax
renormaliza: a atenção que iria para um bloco apagado se redistribui entre os que
ficaram, como no modelo de verdade. O teste test_mascara_igual_a_despejo_fisico
confere a equivalência contra um KV-cache de verdade, apagando os blocos dele.

Fluxo:
    despejos_do_texto(env, i, politica)  ->  quem saiu, em cada unidade, a cada passo
    visibilidade(...)                    ->  bloco-consulta x bloco-chave visível
    with Mascara(modelo, vis, ...):      ->  hooks que trocam a máscara de cada camada
        nll_e_acertos(modelo, ids, ...)
"""

from __future__ import annotations

import copy

import numpy as np

from .llm import _modulos_de_atencao


# ---------------------------------------------------------------- despejos
def despejos_do_texto(env, i: int, politica=None, q_net=None, seed: int | None = 0):
    """Roda a política em todas as unidades do texto i (cada uma é um episódio).

    Passe `politica(env, obs, info)` (as de baselines.py) ou `q_net` (rede Q do DQN,
    chamada uma vez por passo para todas as unidades juntas).
    Devolve (despejos, perdas): despejos[u] é a lista de blocos despejados pela
    unidade u, na ordem; perdas[u] é a atenção perdida média (proxy) dela.
    """
    U = env.U
    envs = [copy.copy(env) for _ in range(U)]
    obs, infos = zip(*[e.reset(seed=seed if u == 0 else None, options={"trace": i, "unidade": u})
                       for u, e in enumerate(envs)])
    obs, infos = np.stack(obs), list(infos)
    despejos = [[] for _ in range(U)]
    perdas = np.zeros(U)
    ativo = np.ones(U, dtype=bool)
    while ativo.any():
        if q_net is not None:
            import torch

            with torch.no_grad():
                q = q_net(torch.as_tensor(obs, dtype=torch.float32, device=next(q_net.parameters()).device))
            acoes = q.argmax(dim=1).cpu().numpy()
        else:
            acoes = [politica(envs[u], obs[u], infos[u]) if ativo[u] else 0 for u in range(U)]
        for u in np.flatnonzero(ativo):
            o, _, term, trunc, info = envs[u].step(int(acoes[u]))
            obs[u], infos[u] = o, info
            despejos[u].append(int(info["bloco_despejado"]))
            if term or trunc:
                ativo[u] = False
                perdas[u] = info["atencao_perdida_media"]
    return despejos, perdas


def visibilidade(nb: int, orcamento: int, despejos: list[int]) -> np.ndarray:
    """(nb, nb) booleano: o bloco-consulta k ainda vê o bloco-chave j?

    Os blocos 0..orcamento entram sem decisão. O despejo do passo p acontece antes
    de o bloco orcamento + 1 + p ser lido, e um bloco despejado nunca volta.
    """
    vis = np.tril(np.ones((nb, nb), dtype=bool))
    for passo, b in enumerate(despejos):
        vis[orcamento + 1 + passo:, b] = False
    return vis


def visibilidade_do_texto(env, despejos: list[list[int]]) -> np.ndarray:
    """(U, nb, nb): uma matriz de visibilidade por unidade (U = 1 no modo geral)."""
    nb = env.traces[0].n_blocos
    return np.stack([visibilidade(nb, env.B, d) for d in despejos])


# ---------------------------------------------------------------- máscara
class Mascara:
    """Enquanto ativa, cada camada de atenção do modelo enxerga só os blocos que a
    política manteve no cache DAQUELA camada (ou daquela cabeça de K/V).

    vis: (U, nb, nb) booleano, com U = 1 (geral: a mesma para todas as camadas),
         U = camadas (por_camada) ou U = camadas x cabeças de K/V (por_cabeca).
    n_tokens: tamanho da sequência no forward. Tokens além de nb * bloco (a resposta
         anexada no fim) contam como parte do último bloco: depois dele não há despejo.
    """

    def __init__(self, modelo, vis: np.ndarray, bloco: int, n_tokens: int, arq):
        import torch

        self.modelo, self.arq = modelo, arq
        self.modulos = _modulos_de_atencao(modelo)
        L, U = len(self.modulos), vis.shape[0]
        if U == 1:
            self.por_camada = lambda l: slice(0, 1)
        elif U == L:
            self.por_camada = lambda l: slice(l, l + 1)
        elif U == L * arq.kv:
            self.por_camada = lambda l: slice(l * arq.kv, (l + 1) * arq.kv)
        else:
            raise ValueError(f"{U} unidades não batem com {L} camadas x {arq.kv} cabeças de K/V")

        dispositivo = next(modelo.parameters()).device
        nb = vis.shape[-1]
        self.vis = torch.as_tensor(vis, device=dispositivo)
        self.idx = torch.clamp(torch.arange(n_tokens, device=dispositivo) // bloco, max=nb - 1)
        self.causal = torch.ones(n_tokens, n_tokens, dtype=torch.bool, device=dispositivo).tril()
        self.camada = {id(m): l for l, m in enumerate(self.modulos)}
        self.ganchos = []

    def mascara_da_camada(self, l: int):
        """(n, T, T) booleano, n = 1 ou o número de cabeças de consulta."""
        v = self.vis[self.por_camada(l)]  # (n_kv ou 1, nb, nb)
        v = v[:, self.idx][:, :, self.idx] & self.causal  # bloco -> token, e causal dentro do bloco
        if v.shape[0] > 1 and self.arq.por_grupo > 1:
            v = v.repeat_interleave(self.arq.por_grupo, dim=0)  # cada consulta usa o cache q // por_grupo
        return v

    def _hook(self, modulo, args, kwargs):
        import torch

        v = self.mascara_da_camada(self.camada[id(modulo)]).unsqueeze(0)  # (1, n, T, T)
        atual = kwargs.get("attention_mask")
        if atual is not None and atual.dtype == torch.bool:
            kwargs["attention_mask"] = v
        else:
            dtype = atual.dtype if atual is not None else next(self.modelo.parameters()).dtype
            m = torch.zeros(v.shape, dtype=dtype, device=v.device)
            kwargs["attention_mask"] = m.masked_fill_(~v, torch.finfo(dtype).min)
        return args, kwargs

    def __enter__(self):
        self.ganchos = [m.register_forward_pre_hook(self._hook, with_kwargs=True) for m in self.modulos]
        return self

    def __exit__(self, *exc):
        for g in self.ganchos:
            g.remove()
        self.ganchos = []


# ---------------------------------------------------------------- métricas
def metricas(nll: np.ndarray, certo: np.ndarray, nll_cheio: np.ndarray, nb: int, bloco: int,
             orcamento: int, n_resposta: int, kl: np.ndarray | None = None) -> dict:
    """Compara um forward com cache podado contra o forward com cache completo.

    Todas as médias de texto são sobre as previsões feitas com o cache já cheio, ou
    seja, a partir do bloco orcamento + 1 até o fim do texto (antes disso nada foi
    despejado e as duas passadas são idênticas).

    kl: KL(p_cheio || p_podado) média por token, em nats. Quanto a distribuição do
        próximo token mudou; é a recompensa densa proposta no Desafio 1. Nunca é negativa.
    delta_nll: aumento médio da NLL por token; exp(delta_nll) - 1 é o aumento relativo
        da perplexidade. Mede só o token que de fato veio, então oscila mais que a KL.
    acerto: a decodificação gulosa ainda produz a senha inteira.
    delta_nll_resposta: aumento da NLL média dos tokens da senha (contínuo, menos ruidoso).
    """
    ini, fim = (orcamento + 1) * bloco, nb * bloco - 1  # posições t que preveem t + 1 dentro do texto
    resp = slice(nb * bloco - 1, nb * bloco - 1 + n_resposta)
    return {
        "kl": float(kl[ini:fim].mean()) if kl is not None else float("nan"),
        "delta_nll": float((nll[ini:fim] - nll_cheio[ini:fim]).mean()),
        "acerto": float(certo[resp].all()) if n_resposta else float("nan"),
        "delta_nll_resposta": float((nll[resp] - nll_cheio[resp]).mean()) if n_resposta else float("nan"),
    }


def spearman(x, y) -> float:
    """Correlação de postos (sem scipy). Empates recebem o posto médio."""
    def postos(v):
        v = np.asarray(v, dtype=float)
        ordem = v.argsort(kind="mergesort")
        r = np.empty(len(v))
        r[ordem] = np.arange(len(v))
        for val in np.unique(v):  # posto médio nos empates
            m = v == val
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    rx, ry = postos(x), postos(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])

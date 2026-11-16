"""Rede Q que dá nota a cada candidato com os mesmos pesos (para o DQN do Stable-Baselines3).

Por que não a MLP padrão
------------------------
A MlpPolicy recebe os 17 candidatos concatenados (17 x 5 números) e devolve 17
valores Q. Ela precisa aprender a mesma regra 17 vezes, uma para cada posição do
vetor, e um exemplo de treino ensina só a posição da ação tomada. No Qwen, ela não
alcançou o H2O em nenhum modo, e no modo geral duas sementes terminaram piores que
a rede antes do treino.

O H2O é uma regra aplicada igual a cada candidato ("despeja quem recebeu menos
atenção"). Esta rede tem a mesma estrutura, mas aprendida:

    e_i = phi(features do candidato i)          mesma rede para os 17
    g   = média dos e_i                          resumo do cache inteiro
    c   = psi(progresso + qual unidade)          contexto
    Q(s, despejar i) = rho([e_i, g, c])          mesma rede para os 17

Ela é equivariante à ordem dos candidatos (trocar dois de lugar troca as notas),
consegue representar o H2O exatamente e tem ~15 mil parâmetros, contra ~30 a 120 mil
da MLP. Continua sendo DQN: só a função Q mudou; ambiente, recompensa, replay buffer,
rede-alvo e exploração são os mesmos (Deep Sets: Zaheer et al., NeurIPS 2017).
"""

from __future__ import annotations

import torch as th
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from torch import nn


class PontuadorPorCandidato(nn.Module):
    """(lote, n_cand * n_feat + contexto) -> (lote, n_cand): um valor Q por candidato."""

    def __init__(self, n_cand: int, n_feat: int, dim_contexto: int, oculta: int = 64):
        super().__init__()
        self.n_cand, self.n_feat = n_cand, n_feat
        self.phi = nn.Sequential(nn.Linear(n_feat, oculta), nn.ReLU(), nn.Linear(oculta, oculta), nn.ReLU())
        self.psi = nn.Sequential(nn.Linear(dim_contexto, oculta), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(3 * oculta, oculta), nn.ReLU(), nn.Linear(oculta, 1))

    def forward(self, x: th.Tensor) -> th.Tensor:
        lote, k = x.shape[0], self.n_cand * self.n_feat
        cand = x[:, :k].reshape(lote, self.n_cand, self.n_feat)
        e = self.phi(cand)  # (lote, n_cand, oculta)
        g = e.mean(dim=1, keepdim=True).expand_as(e)
        c = self.psi(x[:, k:]).unsqueeze(1).expand_as(e)
        return self.rho(th.cat([e, g, c], dim=-1)).squeeze(-1)


class QNetworkPorCandidato(QNetwork):
    """A QNetwork do SB3 com o PontuadorPorCandidato no lugar da MLP."""

    def __init__(self, observation_space, action_space, features_extractor, features_dim: int,
                 n_cand: int, n_feat: int, oculta: int = 64, net_arch=None, activation_fn=nn.ReLU,
                 normalize_images: bool = True):
        super().__init__(observation_space, action_space, features_extractor, features_dim,
                         net_arch=[8], activation_fn=activation_fn, normalize_images=normalize_images)
        if int(action_space.n) != n_cand:
            raise ValueError(f"{action_space.n} ações, mas {n_cand} candidatos")
        self.q_net = PontuadorPorCandidato(n_cand, n_feat, features_dim - n_cand * n_feat, oculta)


class PoliticaPorCandidato(DQNPolicy):
    """DQNPolicy cuja rede Q (e a rede-alvo) é a QNetworkPorCandidato.

    Uso: DQN(PoliticaPorCandidato, env, policy_kwargs=dict(n_cand=17, n_feat=5, oculta=64))
    O DQN.save guarda esses argumentos, e o DQN.load reconstrói a rede sozinho.
    """

    def __init__(self, observation_space, action_space, lr_schedule, n_cand: int, n_feat: int,
                 oculta: int = 64, **kwargs):
        # precisam existir antes do super().__init__, que já monta as redes
        self.n_cand, self.n_feat, self.oculta = n_cand, n_feat, oculta
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def make_q_net(self) -> QNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return QNetworkPorCandidato(n_cand=self.n_cand, n_feat=self.n_feat, oculta=self.oculta,
                                    **net_args).to(self.device)

    def _get_constructor_parameters(self) -> dict:
        data = super()._get_constructor_parameters()
        data.update(n_cand=self.n_cand, n_feat=self.n_feat, oculta=self.oculta)
        return data

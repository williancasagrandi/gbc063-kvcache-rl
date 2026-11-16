"""Rede Q por candidato: formato, equivariância, capacidade de imitar o H2O e salvar/carregar."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from stable_baselines3 import DQN  # noqa: E402

from kvcache_rl import KVCacheEnv, gerar_sintetico  # noqa: E402
from kvcache_rl.baselines import h2o  # noqa: E402
from kvcache_rl.env import FEATURES  # noqa: E402
from kvcache_rl.rede import PoliticaPorCandidato, PontuadorPorCandidato  # noqa: E402

F = len(FEATURES)


@pytest.fixture(scope="module")
def env():
    rng = np.random.default_rng(0)
    tr = [gerar_sintetico(32, 16, rng=rng, nome=f"t{i}", n_camadas=4, n_cabecas=2) for i in range(4)]
    return KVCacheEnv(tr, orcamento=16, modo="por_cabeca")


def test_uma_nota_por_candidato():
    rede = PontuadorPorCandidato(n_cand=17, n_feat=F, dim_contexto=9)
    assert rede(torch.randn(6, 17 * F + 9)).shape == (6, 17)


def test_equivariante_a_ordem_dos_candidatos():
    """Trocar dois candidatos de lugar troca as notas deles e não muda as outras."""
    torch.manual_seed(0)
    rede = PontuadorPorCandidato(n_cand=17, n_feat=F, dim_contexto=3)
    x = torch.randn(4, 17 * F + 3)
    perm = torch.randperm(17)
    cand = x[:, :17 * F].reshape(4, 17, F)
    x2 = torch.cat([cand[:, perm].reshape(4, -1), x[:, 17 * F:]], dim=1)
    assert torch.allclose(rede(x)[:, perm], rede(x2), atol=1e-6)


def test_consegue_imitar_o_h2o(env):
    """A regra do H2O cabe na rede: treinada por supervisão, ela acerta a escolha do H2O."""
    obs, acoes = [], []
    for i in range(len(env.traces)):
        for u in range(env.U):
            o, info = env.reset(options={"trace": i, "unidade": u})
            fim = False
            while not fim:
                a = h2o(env, o, info, n_recentes=4)
                obs.append(o)
                acoes.append(a)
                o, _, fim, _, info = env.step(a)
    x, y = torch.as_tensor(np.array(obs)), torch.as_tensor(acoes)
    torch.manual_seed(0)
    rede = PontuadorPorCandidato(env.n_cand, F, x.shape[1] - env.n_cand * F)
    opt = torch.optim.Adam(rede.parameters(), lr=3e-3)
    for _ in range(400):
        perda = torch.nn.functional.cross_entropy(rede(x), y)
        opt.zero_grad()
        perda.backward()
        opt.step()
    acerto = (rede(x).argmax(1) == y).float().mean().item()
    assert acerto > 0.95, acerto


def test_dqn_salva_e_carrega(env, tmp_path):
    kw = dict(n_cand=env.n_cand, n_feat=F, oculta=32)
    m = DQN(PoliticaPorCandidato, env, policy_kwargs=kw, learning_starts=50, seed=0, device="cpu")
    m.learn(200)
    assert type(m.policy.q_net_target.q_net).__name__ == "PontuadorPorCandidato"  # a rede-alvo também
    m.save(tmp_path / "m")
    m2 = DQN.load(tmp_path / "m", device="cpu")
    obs, _ = env.reset(options={"trace": 0, "unidade": 5})
    x = torch.as_tensor(obs[None])
    assert torch.allclose(m.policy.q_net(x), m2.policy.q_net(x))
    assert m2.policy.q_net(x).shape == (1, env.n_cand)

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from kvcache_rl import KVCacheEnv, gerar_sintetico
from kvcache_rl.baselines import BASELINES, avaliar


@pytest.fixture(scope="module")
def traces():
    rng = np.random.default_rng(123)
    return [gerar_sintetico(40, 8, rng=rng, nome=f"t{i}") for i in range(6)]


def test_trace_causal_e_normalizado(traces):
    tr = traces[0]
    assert np.allclose(tr.att.sum(axis=1), 1.0, atol=1e-4)
    for t in range(tr.n_tokens):
        assert np.all(tr.att[t, t // tr.bloco + 1:] == 0)


def test_check_env(traces):
    check_env(KVCacheEnv(traces, orcamento=8), skip_render_check=True)


def test_cache_nunca_estoura(traces):
    env = KVCacheEnv(traces, orcamento=8)
    obs, info = env.reset(seed=0)
    passos, fim = 0, False
    while not fim:
        assert len(env.cache) == env.B + 1
        obs, r, fim, _, info = env.step(env.action_space.sample())
        assert r <= env.bonus_agulha + 1e-9
        passos += 1
    assert passos == env.nb - 1 - env.B


def test_candidatos_ordenados(traces):
    env = KVCacheEnv(traces, orcamento=8)
    _, info = env.reset(seed=1)
    for _ in range(5):
        assert np.all(np.diff(info["candidatos"]) > 0)
        assert info["candidatos"][-1] == env.k
        _, _, _, _, info = env.step(3)


@pytest.mark.parametrize("nome", list(BASELINES))
def test_baselines_rodam(traces, nome):
    env = KVCacheEnv(traces, orcamento=8)
    r = avaliar(env, BASELINES[nome])
    assert 0.0 <= r["atencao_perdida"] <= 1.0


def test_agulha_gpt2_fora_do_sink_e_no_bloco_certo():
    from kvcache_rl.gpt2 import montar_sequencia

    rng = np.random.default_rng(0)
    corpo = list(range(1000, 3000))
    prefixo, codigo, ponto, pergunta = [1, 2, 3], [7, 7], [4], [5, 6, 8]
    for _ in range(200):
        s = montar_sequencia(corpo, prefixo, codigo, ponto, pergunta, 1024, 16, rng)
        assert len(s.ids) == 1024
        assert 2 <= s.bloco_agulha < 1024 // 16 // 4 + 1
        i = s.ids.index(7)  # primeiro token do código
        assert s.ids[i + 1] == 7 and (i + 1) // 16 == s.bloco_agulha
        assert s.ids[-3:] == pergunta


def test_oraculo_e_limite_superior(traces):
    env = KVCacheEnv(traces, orcamento=8)
    perdas = {n: avaliar(env, BASELINES[n])["atencao_perdida"] for n in ("oraculo", "janela_deslizante", "aleatoria")}
    assert perdas["oraculo"] <= perdas["janela_deslizante"]
    assert perdas["oraculo"] <= perdas["aleatoria"]

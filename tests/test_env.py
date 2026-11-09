import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from kvcache_rl import MODOS, KVCacheEnv, agrupar_camadas, gerar_sintetico
from kvcache_rl.baselines import BASELINES, avaliar

CONFIGS = [("geral", 1), ("grupos", 3), ("por_camada", 3), ("por_camada", 12)]


@pytest.fixture(scope="module")
def traces():
    rng = np.random.default_rng(123)
    return [gerar_sintetico(40, 8, rng=rng, nome=f"t{i}") for i in range(6)]


def test_trace_causal_e_normalizado(traces):
    tr = traces[0]
    assert np.allclose(tr.att.sum(axis=1), 1.0, atol=1e-4)
    for t in range(tr.n_tokens):
        assert np.all(tr.att[t, t // tr.bloco + 1:] == 0)


def test_camadas_somam_um_e_media_bate(traces):
    tr = traces[0]
    assert tr.camadas.shape == (12, tr.n_tokens, tr.n_blocos)
    assert np.allclose(tr.camadas.astype(np.float32).sum(axis=2), 1.0, atol=2e-2)
    assert np.allclose(tr.camadas.astype(np.float32).mean(axis=0), tr.att, atol=1e-3)
    assert np.allclose(agrupar_camadas(tr, 1)[0], tr.att, atol=1e-3)


def test_camadas_sao_diferentes(traces):
    """Cada camada precisa olhar para coisas diferentes, senão por_camada não faz sentido."""
    c = traces[0].camadas.astype(np.float32)
    assert np.abs(c[0] - c[-1]).mean() > 0.005


@pytest.mark.parametrize("modo,grupos", CONFIGS)
def test_check_env(traces, modo, grupos):
    check_env(KVCacheEnv(traces, orcamento=8, modo=modo, grupos=grupos), skip_render_check=True)


@pytest.mark.parametrize("modo,grupos", CONFIGS)
def test_cache_nunca_estoura_e_tamanho_do_episodio(traces, modo, grupos):
    env = KVCacheEnv(traces, orcamento=8, modo=modo, grupos=grupos)
    env.reset(seed=0)
    passos, fim = 0, False
    while not fim:
        assert len(env.cache) == env.B + 1
        _, r, fim, _, info = env.step(env.action_space.sample())
        assert r <= env.bonus_agulha + 1e-9
        passos += 1
    assert passos == env.nb - 1 - env.B
    assert info["agulha_mantida"] in (0.0, 1.0)


def test_candidatos_ordenados(traces):
    env = KVCacheEnv(traces, orcamento=8, modo="por_camada", grupos=3)
    _, info = env.reset(seed=1)
    for _ in range(9):
        assert np.all(np.diff(info["candidatos"]) > 0)
        assert info["candidatos"][-1] == env.k
        _, _, _, _, info = env.step(3)


def test_por_camada_episodio_e_uma_camada(traces):
    env = KVCacheEnv(traces, orcamento=8, modo="por_camada", grupos=12)
    obs, info = env.reset(options={"trace": 2, "camada": 7})
    assert info["camada"] == 7 and obs[-12:].argmax() == 7 and obs[-12:].sum() == 1
    assert len(env.episodios_avaliacao()) == len(traces) * 12
    assert env.decisoes_por_texto == (env.nb - 1 - env.B) * 12


@pytest.mark.parametrize("modo", MODOS)
@pytest.mark.parametrize("nome", list(BASELINES))
def test_baselines_rodam(traces, modo, nome):
    env = KVCacheEnv(traces, orcamento=8, modo=modo, grupos=3)
    r = avaliar(env, BASELINES[nome])
    assert 0.0 <= r["atencao_perdida"] <= 1.0


@pytest.mark.parametrize("nome", ["janela_deslizante", "streaming_llm", "h2o"])
def test_grupos_nao_muda_as_regras_fixas(traces, nome):
    """No modo grupos o cache é um só: as regras fixas decidem igual ao modo geral."""
    geral = avaliar(KVCacheEnv(traces, orcamento=8, modo="geral"), BASELINES[nome])
    grupos = avaliar(KVCacheEnv(traces, orcamento=8, modo="grupos", grupos=3), BASELINES[nome])
    assert grupos["atencao_perdida"] == pytest.approx(geral["atencao_perdida"], abs=2e-3)
    assert grupos["agulha"] == geral["agulha"]


def test_por_camada_com_um_grupo_equivale_ao_geral(traces):
    geral = avaliar(KVCacheEnv(traces, orcamento=8, modo="geral"), BASELINES["h2o"])
    um = avaliar(KVCacheEnv(traces, orcamento=8, modo="por_camada", grupos=1), BASELINES["h2o"])
    assert um["atencao_perdida"] == pytest.approx(geral["atencao_perdida"], abs=2e-3)


def test_h2o_por_camada_perde_menos_que_h2o_geral(traces):
    """Decidir por camada usa a atenção de cada uma: o H2O deve perder menos."""
    geral = avaliar(KVCacheEnv(traces, orcamento=8, modo="geral"), BASELINES["h2o"])
    camada = avaliar(KVCacheEnv(traces, orcamento=8, modo="por_camada", grupos=12), BASELINES["h2o"])
    assert camada["atencao_perdida"] < geral["atencao_perdida"]


@pytest.mark.parametrize("modo,grupos", CONFIGS)
def test_oraculo_e_limite_superior(traces, modo, grupos):
    env = KVCacheEnv(traces, orcamento=8, modo=modo, grupos=grupos)
    perdas = {n: avaliar(env, BASELINES[n])["atencao_perdida"] for n in ("oraculo", "janela_deslizante", "aleatoria")}
    assert perdas["oraculo"] <= perdas["janela_deslizante"]
    assert perdas["oraculo"] <= perdas["aleatoria"]
    assert avaliar(env, BASELINES["oraculo"])["agulha"] == 1.0


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

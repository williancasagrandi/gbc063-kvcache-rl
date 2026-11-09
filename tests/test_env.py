import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from kvcache_rl import MODOS, KVCacheEnv, Trace, agregar_blocos, gerar_sintetico
from kvcache_rl.baselines import BASELINES, avaliar


@pytest.fixture(scope="module")
def traces():
    rng = np.random.default_rng(123)
    return [gerar_sintetico(40, 8, rng=rng, nome=f"t{i}", n_camadas=6, n_cabecas=4) for i in range(6)]


def test_trace_causal_e_normalizado(traces):
    tr = traces[0]
    for m in (tr.att, tr.camadas.astype(np.float32), tr.cabecas.astype(np.float32)):
        assert np.allclose(m.sum(axis=-1), 1.0, atol=2e-2)
        assert np.all(np.triu(m, k=1) == 0)


def test_medias_batem(traces):
    tr = traces[0]
    cab = tr.cabecas.astype(np.float32)
    assert np.allclose(cab.mean(axis=1), tr.camadas.astype(np.float32), atol=2e-3)
    assert np.allclose(tr.camadas.astype(np.float32).mean(axis=0), tr.att, atol=2e-3)


def test_cabecas_sao_diferentes(traces):
    cab = traces[0].cabecas.astype(np.float32).reshape(-1, 40, 40)
    assert np.abs(cab[0] - cab[-1]).mean() > 0.005


def test_agregar_blocos():
    rng = np.random.default_rng(0)
    T, bs = 32, 4
    a = np.tril(rng.random((T, T)))
    a /= a.sum(axis=1, keepdims=True)
    b = agregar_blocos(a, bs)
    assert b.shape == (8, 8)
    assert np.allclose(b.sum(axis=1), 1.0)
    assert np.all(np.triu(b, k=1) == 0)


def test_ambiente_por_bloco_igual_a_simulacao_por_token():
    """A perda do ambiente (por bloco) tem que ser igual à calculada token a token."""
    rng = np.random.default_rng(7)
    nb, bs, B = 24, 4, 6
    T = nb * bs
    a = np.tril(rng.random((T, T)) ** 3)
    a /= a.sum(axis=1, keepdims=True)
    tr = Trace(att=agregar_blocos(a, bs).astype(np.float32), bloco=bs, agulha=-1)
    env = KVCacheEnv([tr], orcamento=B)
    env.reset(options={"trace": 0, "unidade": 0})

    cache, despejados, perdas = list(range(B + 1)), set(), []
    fim, passo, k = False, 0, B
    while not fim:
        acao = passo % (B + 1)
        despejados.add(cache.pop(acao))
        k += 1
        cache.append(k)
        tokens_desp = [s for j in despejados for s in range(j * bs, (j + 1) * bs)]
        perdas.append(np.mean([a[t, tokens_desp].sum() for t in range(k * bs, (k + 1) * bs)]))
        _, _, fim, _, info = env.step(acao)
        passo += 1
    assert info["atencao_perdida_media"] == pytest.approx(np.mean(perdas), abs=1e-5)


@pytest.mark.parametrize("modo", MODOS)
def test_check_env(traces, modo):
    check_env(KVCacheEnv(traces, orcamento=8, modo=modo), skip_render_check=True)


@pytest.mark.parametrize("modo", MODOS)
def test_cache_nunca_estoura_e_tamanho_do_episodio(traces, modo):
    env = KVCacheEnv(traces, orcamento=8, modo=modo)
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
    env = KVCacheEnv(traces, orcamento=8, modo="por_cabeca")
    _, info = env.reset(seed=1)
    for _ in range(9):
        assert np.all(np.diff(info["candidatos"]) > 0)
        assert info["candidatos"][-1] == env.k
        _, _, _, _, info = env.step(3)


def test_unidades(traces):
    camada = KVCacheEnv(traces, orcamento=8, modo="por_camada")
    cabeca = KVCacheEnv(traces, orcamento=8, modo="por_cabeca")
    assert (camada.U, cabeca.U) == (6, 24)
    obs, info = cabeca.reset(options={"trace": 2, "unidade": 13})
    assert info["unidade"] == 13 and obs[-24:].argmax() == 13 and obs[-24:].sum() == 1
    assert cabeca.nome_unidade(13) == "camada 3, cabeça 1"
    assert len(cabeca.episodios_avaliacao()) == len(traces) * 24


@pytest.mark.parametrize("modo", MODOS)
def test_pesos_da_agulha(traces, modo):
    env = KVCacheEnv(traces, orcamento=8, modo=modo)
    for rel, peso in zip(env._relevancia, env._peso):
        assert rel.max() == pytest.approx(1.0) and peso.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("modo", MODOS)
@pytest.mark.parametrize("nome", list(BASELINES))
def test_baselines_rodam(traces, modo, nome):
    r = avaliar(KVCacheEnv(traces, orcamento=8, modo=modo), BASELINES[nome])
    assert 0.0 <= r["atencao_perdida"] <= 1.0
    assert 0.0 <= r["agulha"] <= 1.0


def test_decidir_mais_fino_perde_menos(traces):
    """Com a atenção de cada unidade, o H2O perde menos: cabeça < camada < geral."""
    perda = {m: avaliar(KVCacheEnv(traces, orcamento=8, modo=m), BASELINES["h2o"])["atencao_perdida"] for m in MODOS}
    assert perda["por_cabeca"] < perda["por_camada"] < perda["geral"]


@pytest.mark.parametrize("modo", MODOS)
def test_oraculo_e_limite_superior(traces, modo):
    env = KVCacheEnv(traces, orcamento=8, modo=modo)
    perdas = {n: avaliar(env, BASELINES[n])["atencao_perdida"] for n in ("oraculo", "janela_deslizante", "aleatoria")}
    assert perdas["oraculo"] <= perdas["janela_deslizante"]
    assert perdas["oraculo"] <= perdas["aleatoria"]
    assert avaliar(env, BASELINES["oraculo"])["agulha"] == pytest.approx(1.0)


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

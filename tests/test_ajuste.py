"""O ajuste das regras fixas é justo: escolhe pelo critério do DQN, só com textos de treino."""

import json

import numpy as np
import pytest

from kvcache_rl import KVCacheEnv, gerar_sintetico
from kvcache_rl.baselines import (BASELINES, PARAMETROS, ajustar, avaliar, com_parametros, grade_de, h2o,
                                  politicas_ajustadas, streaming_llm)


@pytest.fixture(scope="module")
def traces():
    rng = np.random.default_rng(7)
    return [gerar_sintetico(32, 16, rng=rng, nome=f"t{i}", n_camadas=3, n_cabecas=2) for i in range(4)]


def test_h2o_sem_recentes_nao_protege_todos(traces):
    """score[-0:] é o vetor inteiro: sem o cuidado, n_recentes=0 viraria 'despeja o índice 0'."""
    env = KVCacheEnv(traces, orcamento=8)
    _, info = env.reset(options={"trace": 0})
    score = info["atencao_acumulada"]
    assert h2o(env, None, info, n_recentes=0) == int(np.argmin(score))


def test_h2o_protege_os_recentes(traces):
    env = KVCacheEnv(traces, orcamento=8)
    _, info = env.reset(options={"trace": 1})
    fim = False
    while not fim:
        a = h2o(env, None, info, n_recentes=3)
        assert a < env.n_cand - 3  # nunca um dos 3 mais recentes
        _, _, fim, _, info = env.step(a)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_streaming_llm_mantem_os_sinks(traces, n):
    env = KVCacheEnv(traces, orcamento=8)
    _, info = env.reset(options={"trace": 0})
    fim = False
    while not fim:
        _, _, fim, _, info = env.step(streaming_llm(env, None, info, n_sink=n))
    assert set(range(n)) <= set(env.cache)


@pytest.mark.parametrize("modo", ["geral", "por_cabeca"])
def test_ajuste_escolhe_o_minimo_da_grade(traces, modo):
    env = KVCacheEnv(traces, orcamento=8, modo=modo)
    for nome in PARAMETROS:
        r = ajustar(env, nome)
        assert len(r["grade"]) == len(next(iter(grade_de(nome, env).values())))
        melhor = min(g["atencao_perdida"] for g in r["grade"])
        escolhido = avaliar(env, com_parametros(nome, r["melhor"]))["atencao_perdida"]
        assert escolhido == pytest.approx(melhor)


def test_ajuste_nao_olha_o_teste(traces):
    """O resultado do ajuste só depende dos textos de treino passados ao ambiente."""
    treino = KVCacheEnv(traces[:2], orcamento=8)
    antes = ajustar(treino, "h2o")
    _ = KVCacheEnv(traces[2:], orcamento=8)  # o teste existe, mas não entra
    assert ajustar(KVCacheEnv(traces[:2], orcamento=8), "h2o") == antes


def test_politicas_ajustadas(tmp_path, traces):
    caminho = tmp_path / "ajuste.json"
    pols, achou = politicas_ajustadas(caminho)
    assert not achou and pols["h2o"] is BASELINES["h2o"]
    caminho.write_text(json.dumps({"h2o": {"melhor": {"n_recentes": 2}, "grade": []}}))
    pols, achou = politicas_ajustadas(caminho)
    assert achou and pols["h2o"].keywords == {"n_recentes": 2}
    assert pols["janela_deslizante"] is BASELINES["janela_deslizante"]


def test_grade_do_h2o_cobre_todo_o_orcamento(traces):
    """O ótimo não pode ficar fora da grade: de 0 recentes protegidos até o orçamento inteiro."""
    env = KVCacheEnv(traces, orcamento=8)
    assert grade_de("h2o", env) == {"n_recentes": list(range(9))}


def test_tova_despeja_quem_teve_menos_atencao_recente(traces):
    from kvcache_rl.baselines import tova

    env = KVCacheEnv(traces, orcamento=8, modo="por_cabeca")
    _, info = env.reset(options={"trace": 0, "unidade": 3})
    fim = False
    while not fim:
        a = tova(env, None, info)
        assert info["atencao_recente"][a] == info["atencao_recente"].min()
        _, _, fim, _, info = env.step(a)

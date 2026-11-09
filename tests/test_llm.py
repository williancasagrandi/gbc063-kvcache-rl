"""Testes da gravação de traces que não precisam baixar modelo (sem torch/transformers)."""

import numpy as np
import pytest

from kvcache_rl.llm import MODELOS, Arquitetura, agrupar_kv, montar_sequencia


def test_atalhos_de_modelo():
    for chave, m in MODELOS.items():
        assert m["modelo"] and m["tokens"] % 16 == 0, chave


@pytest.mark.parametrize("consultas,kv,por_grupo", [(12, 12, 1), (14, 2, 7), (32, 8, 4)])
def test_arquitetura(consultas, kv, por_grupo):
    arq = Arquitetura(camadas=24, consultas=consultas, kv=kv, max_tokens=4096)
    assert arq.por_grupo == por_grupo
    assert arq.unidades == 24 * kv


def test_agrupar_kv_sem_gqa_nao_muda_nada():
    rng = np.random.default_rng(0)
    att = rng.random((12, 8, 8))
    assert agrupar_kv(att, 1) is att


def test_agrupar_kv_media_o_grupo_certo():
    # repeat_kv repete cada cabeça de K/V em sequência: a consulta q usa o cache q // por_grupo
    att = np.arange(14 * 2 * 2, dtype=np.float64).reshape(14, 2, 2)
    g = agrupar_kv(att, 7)
    assert g.shape == (2, 2, 2)
    assert np.allclose(g[0], att[:7].mean(axis=0))
    assert np.allclose(g[1], att[7:].mean(axis=0))


def test_agrupar_kv_preserva_normalizacao():
    rng = np.random.default_rng(1)
    att = rng.random((14, 6, 6))
    att /= att.sum(axis=-1, keepdims=True)
    g = agrupar_kv(att, 7)
    assert np.allclose(g.sum(axis=-1), 1.0)


def test_agrupar_kv_recusa_grupo_invalido():
    with pytest.raises(ValueError):
        agrupar_kv(np.zeros((14, 4, 4)), 4)


def test_agulha_fora_do_sink_e_no_bloco_certo():
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

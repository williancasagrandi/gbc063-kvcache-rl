"""Testes da gravação da atenção contra modelos reais minúsculos, criados com pesos aleatórios.

Não baixa nada: monta um Qwen2 (com GQA) e um GPT-2 (sem GQA) de brinquedo a partir
da configuração e confere formato, normalização, causalidade e o agrupamento por
cabeça de K/V, que é a parte que muda de uma arquitetura para a outra.
"""

import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from kvcache_rl.llm import (Sequencia, _Gravador, agrupar_kv, atencao_por_cabeca,  # noqa: E402
                            ler_arquitetura, responde_certo)
from kvcache_rl.traces import agregar_blocos  # noqa: E402

TOKENS, BLOCO = 64, 8
IDS = list(np.random.default_rng(0).integers(0, 200, TOKENS))


def qwen_de_brinquedo(n_camadas=4, consultas=14, kv=2):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=200, hidden_size=8 * consultas, intermediate_size=128,
                      num_hidden_layers=n_camadas, num_attention_heads=consultas, num_key_value_heads=kv,
                      max_position_embeddings=256, attn_implementation="eager")
    return Qwen2ForCausalLM(cfg).eval()


def gpt2_de_brinquedo(n_camadas=3, cabecas=6):
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=200, n_positions=256, n_embd=16 * cabecas, n_layer=n_camadas,
                     n_head=cabecas, attn_implementation="eager")
    return GPT2LMHeadModel(cfg).eval()


def test_repeat_kv_confirma_o_agrupamento():
    """A premissa do agrupar_kv: a cabeça de consulta q usa o cache q // por_grupo."""
    from transformers.models.qwen2.modeling_qwen2 import repeat_kv

    x = torch.arange(2 * 5, dtype=torch.float32).reshape(1, 2, 5, 1)  # (lote, cabeças de K/V, tokens, dim)
    r = repeat_kv(x, 7)
    assert r.shape[1] == 14
    assert all(torch.equal(r[0, q], x[0, q // 7]) for q in range(14))


def test_arquitetura_lida_dos_modelos():
    assert ler_arquitetura(qwen_de_brinquedo()).por_grupo == 7
    assert ler_arquitetura(gpt2_de_brinquedo()).por_grupo == 1  # MHA: cada consulta tem o seu K/V


@pytest.mark.parametrize("fabrica,por_grupo,forma", [
    (qwen_de_brinquedo, 7, (4, 2)),   # GQA, como o Qwen2.5-0.5B
    (gpt2_de_brinquedo, 1, (3, 6)),   # MHA, como o GPT-2
])
def test_atencao_por_cabeca(fabrica, por_grupo, forma):
    modelo = fabrica()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # o hook tem de funcionar: o fallback gasta memória
        att = atencao_por_cabeca(modelo, IDS, BLOCO, "cpu", por_grupo)

    nb = TOKENS // BLOCO
    assert att.shape == (*forma, nb, nb)
    assert np.allclose(att.sum(axis=-1), 1.0, atol=1e-5)  # cada linha é uma distribuição
    assert (np.triu(att, 1) == 0).all()  # causal: ninguém olha para o futuro
    assert att.min() >= 0.0


def test_grupo_de_consultas_vai_para_a_cabeca_de_kv_certa():
    """Zerando a projeção Q das 7 primeiras consultas, a atenção delas fica uniforme
    (todos os scores viram 0). Se o agrupamento estiver certo, só a cabeça de K/V 0 fica uniforme."""
    modelo = qwen_de_brinquedo(n_camadas=1)
    at = modelo.model.layers[0].self_attn
    fim = 7 * at.head_dim
    with torch.no_grad():
        at.q_proj.weight[:fim].zero_()
        at.q_proj.bias[:fim].zero_()
        # as outras 7 consultas ganham scores grandes, para a atenção delas ficar bem longe da uniforme
        at.q_proj.weight[fim:] *= 50.0

    att = atencao_por_cabeca(modelo, IDS, BLOCO, "cpu", por_grupo=7)[0]
    # o esperado é a atenção uniforme token a token (1/(i+1) para j <= i) agregada por bloco
    por_token = np.tril(np.ones((TOKENS, TOKENS))) / np.arange(1, TOKENS + 1)[:, None]
    uniforme = agregar_blocos(por_token, BLOCO)
    assert np.allclose(att[0], uniforme, atol=1e-4)
    assert not np.allclose(att[1], uniforme, atol=1e-2)


def test_hook_tira_a_matriz_grande_da_saida():
    g = _Gravador(bloco=2, por_grupo=1)
    pesos = torch.rand(1, 3, 4, 4)
    saida = g.hook(None, None, ("saida_da_atencao", pesos))
    assert saida == ("saida_da_atencao", None)  # a matriz saiu: o PyTorch libera na hora
    assert len(g.camadas) == 1 and g.camadas[0].shape == (3, 2, 2)


def test_responde_certo():
    modelo = qwen_de_brinquedo(n_camadas=1)

    class TokFalso:  # dublê: não precisamos de um tokenizador baixado para testar a comparação
        eos_token_id = 0

        def decode(self, ids, skip_special_tokens=True):
            return " 1234"

    ok, resposta = responde_certo(modelo, TokFalso(), Sequencia(ids=IDS, bloco_agulha=2, codigo="1234"), "cpu", 2)
    assert ok and resposta == "1234"
    assert not responde_certo(modelo, TokFalso(), Sequencia(ids=IDS, bloco_agulha=2, codigo="9999"), "cpu", 2)[0]


def test_agrupar_kv_bate_com_a_media_manual():
    rng = np.random.default_rng(3)
    att = rng.random((14, 4, 4))
    g = agrupar_kv(att, 7)
    for kv in range(2):
        esperado = np.mean([att[q] for q in range(14) if q // 7 == kv], axis=0)
        assert np.allclose(g[kv], esperado)

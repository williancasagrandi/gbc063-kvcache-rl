"""A avaliação no modelo real mede o que diz medir?

O teste central compara duas formas de calcular a mesma coisa num Qwen2 (GQA) e num
GPT-2 de brinquedo: (a) ler o texto bloco a bloco com um KV-cache de verdade,
apagando dele os blocos que a política despeja; (b) um único forward com a máscara
que modelo_real.Mascara monta. Se as duas derem os mesmos logits, a avaliação com
máscara é um despejo de verdade, e não uma aproximação.
"""

import numpy as np
import pytest

from kvcache_rl import KVCacheEnv, gerar_sintetico
from kvcache_rl.baselines import BASELINES, avaliar
from kvcache_rl.modelo_real import despejos_do_texto, metricas, spearman, visibilidade, visibilidade_do_texto

NB, BLOCO, B = 10, 4, 3
T = NB * BLOCO


# ------------------------------------------------------------- sem modelo
def test_visibilidade_bate_com_o_cache_do_ambiente():
    """A linha k da visibilidade é exatamente o cache que o ambiente tinha ao ler o bloco k."""
    rng = np.random.default_rng(0)
    tr = [gerar_sintetico(24, 16, rng=rng, nome="t")]
    env = KVCacheEnv(tr, orcamento=8)
    env.reset(options={"trace": 0})
    saiu, caches = [], {}
    fim = False
    while not fim:
        _, _, fim, _, info = env.step(int(rng.integers(env.n_cand)))
        saiu.append(info["bloco_despejado"])
        caches[env.k] = sorted(env.cache)
    vis = visibilidade(24, 8, saiu)
    for k, cache in caches.items():
        assert np.flatnonzero(vis[k]).tolist() == cache
    assert vis[:9].sum() == np.tril(np.ones((9, 9))).sum()  # antes de encher, vê tudo


@pytest.mark.parametrize("modo", ["geral", "por_camada", "por_cabeca"])
def test_despejos_do_texto_reproduz_a_avaliacao_do_proxy(modo):
    rng = np.random.default_rng(1)
    tr = [gerar_sintetico(24, 16, rng=rng, nome=f"t{i}", n_camadas=4, n_cabecas=3) for i in range(3)]
    env = KVCacheEnv(tr, orcamento=8, modo=modo)
    perdas = [despejos_do_texto(env, i, BASELINES["h2o"])[1].mean() for i in range(3)]
    assert np.mean(perdas) == pytest.approx(avaliar(env, BASELINES["h2o"])["atencao_perdida"])
    despejos, _ = despejos_do_texto(env, 0, BASELINES["h2o"])
    assert visibilidade_do_texto(env, despejos).shape == (env.U, 24, 24)


def test_metricas():
    cheio = np.zeros(T - 1 + 2)
    podado = cheio.copy()
    podado[(B + 1) * BLOCO:T - 1] = 0.5  # texto piora 0,5 nat por token depois que o cache enche
    certo = np.ones(T - 1 + 2, dtype=bool)
    m = metricas(podado, certo, cheio, NB, BLOCO, B, 2)
    assert m["delta_nll"] == pytest.approx(0.5) and m["acerto"] == 1.0 and m["delta_nll_resposta"] == 0.0
    certo[T - 1] = False  # erra o primeiro token da senha
    assert metricas(podado, certo, cheio, NB, BLOCO, B, 2)["acerto"] == 0.0


def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2, 3], [1, 1, 1]) != spearman([1, 2, 3], [1, 1, 1])  # nan sem variação


# ------------------------------------------------------------- com modelo
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from kvcache_rl.llm import ler_arquitetura  # noqa: E402
from kvcache_rl.modelo_real import Mascara  # noqa: E402

IDS = np.random.default_rng(0).integers(0, 200, T).tolist()


def qwen(n_camadas=3):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    return Qwen2ForCausalLM(Qwen2Config(
        vocab_size=200, hidden_size=112, intermediate_size=128, num_hidden_layers=n_camadas,
        num_attention_heads=14, num_key_value_heads=2, max_position_embeddings=256,
        attn_implementation="eager")).eval()


def gpt2(n_camadas=3):
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(vocab_size=200, n_positions=256, n_embd=64, n_layer=n_camadas,
                                      n_head=4, attn_implementation="eager")).eval()


def escalonamento_aleatorio(rng) -> list[int]:
    """Um despejo válido por passo: sempre um bloco que está no cache naquele momento."""
    cache, saiu = list(range(B + 1)), []
    for k in range(B + 1, NB):
        b = int(rng.choice(cache))
        cache.remove(b)
        saiu.append(b)
        cache.append(k)
    return saiu


def logits_com_despejo_fisico(modelo, esc_por_camada: list[list[int]]) -> torch.Tensor:
    """Lê o texto bloco a bloco com um KV-cache de verdade e APAGA dele os blocos despejados."""
    from transformers import DynamicCache

    from kvcache_rl.llm import _modulos_de_atencao

    modulos = _modulos_de_atencao(modelo)
    camada = {id(m): l for l, m in enumerate(modulos)}
    cache, posicoes = DynamicCache(), [[] for _ in modulos]

    def mascara(modulo, args, kwargs):
        # o bloco novo vê tudo que sobrou no cache desta camada + ele mesmo, causalmente
        antes = len(posicoes[camada[id(modulo)]])
        m = torch.zeros(1, 1, BLOCO, antes + BLOCO)
        m[0, 0, :, antes:] = torch.full((BLOCO, BLOCO), torch.finfo(torch.float32).min).triu(1)
        kwargs["attention_mask"] = m
        return args, kwargs

    ganchos = [m.register_forward_pre_hook(mascara, with_kwargs=True) for m in modulos]
    saidas = []
    try:
        with torch.no_grad():
            for k in range(NB):
                pos = torch.arange(k * BLOCO, (k + 1) * BLOCO)
                out = modelo(torch.tensor([IDS[k * BLOCO:(k + 1) * BLOCO]]), position_ids=pos.unsqueeze(0),
                             past_key_values=cache, use_cache=True)
                saidas.append(out.logits[0])
                for l in range(len(modulos)):
                    posicoes[l].extend(pos.tolist())
                if B <= k < NB - 1:  # despejo antes do próximo bloco: apaga K e V da memória
                    for l, esc in enumerate(esc_por_camada):
                        fora = set(range(esc[k - B] * BLOCO, (esc[k - B] + 1) * BLOCO))
                        fica = torch.tensor([i for i, p in enumerate(posicoes[l]) if p not in fora])
                        lay = cache.layers[l]
                        lay.keys, lay.values = lay.keys[:, :, fica], lay.values[:, :, fica]
                        posicoes[l] = [p for p in posicoes[l] if p not in fora]
    finally:
        for g in ganchos:
            g.remove()
    return torch.cat(saidas)


def logits_com_mascara(modelo, vis: np.ndarray) -> torch.Tensor:
    with Mascara(modelo, vis, BLOCO, T, ler_arquitetura(modelo)), torch.no_grad():
        return modelo(torch.tensor([IDS])).logits[0]


@pytest.mark.parametrize("fabrica", [qwen, gpt2])
def test_mascara_igual_a_despejo_fisico_geral(fabrica):
    modelo = fabrica()
    esc = escalonamento_aleatorio(np.random.default_rng(3))
    fisico = logits_com_despejo_fisico(modelo, [esc] * 3)
    mascarado = logits_com_mascara(modelo, visibilidade(NB, B, esc)[None])
    assert torch.allclose(fisico, mascarado, atol=1e-4)
    # e o despejo de fato muda a saída (senão o teste não provaria nada)
    with torch.no_grad():
        cheio = modelo(torch.tensor([IDS])).logits[0]
    assert not torch.allclose(cheio[(B + 1) * BLOCO:], mascarado[(B + 1) * BLOCO:], atol=1e-3)


@pytest.mark.parametrize("fabrica", [qwen, gpt2])
def test_mascara_igual_a_despejo_fisico_por_camada(fabrica):
    """Cada camada apaga blocos diferentes do seu próprio cache."""
    modelo = fabrica()
    rng = np.random.default_rng(4)
    esc = [escalonamento_aleatorio(rng) for _ in range(3)]
    fisico = logits_com_despejo_fisico(modelo, esc)
    mascarado = logits_com_mascara(modelo, np.stack([visibilidade(NB, B, e) for e in esc]))
    assert torch.allclose(fisico, mascarado, atol=1e-4)


@pytest.mark.parametrize("fabrica", [qwen, gpt2])
def test_kl_contra_o_cache_completo(fabrica):
    """KL = 0 sem despejo; com despejo é > 0 só depois que o cache enche, e bate com a conta direta."""
    from kvcache_rl.llm import passada

    modelo = fabrica()
    cheio = passada(modelo, IDS, "cpu")
    assert np.allclose(passada(modelo, IDS, "cpu", referencia=cheio["h"])["kl"], 0, atol=1e-6)

    esc = escalonamento_aleatorio(np.random.default_rng(5))
    with Mascara(modelo, visibilidade(NB, B, esc)[None], BLOCO, T, ler_arquitetura(modelo)):
        r = passada(modelo, IDS, "cpu", referencia=cheio["h"], pedaco=7)
    kl = r["kl"]
    assert np.allclose(kl[:(B + 1) * BLOCO], 0, atol=1e-6)  # antes de encher, nada foi despejado
    assert kl[(B + 1) * BLOCO:].mean() > 1e-4 and (kl >= 0).all()

    with torch.no_grad():  # conta direta, com os logits completos
        p_cheio = torch.log_softmax(modelo(torch.tensor([IDS])).logits[0, :-1].float(), -1)
    podado = torch.log_softmax(logits_com_mascara(modelo, visibilidade(NB, B, esc)[None])[:-1].float(), -1)
    direto = (p_cheio.exp() * (p_cheio - podado)).sum(-1).numpy()
    assert np.allclose(kl, direto, atol=1e-5)

    m = metricas(r["nll"], r["certo"], cheio["nll"], NB, BLOCO, B, 0, kl)
    assert m["kl"] == pytest.approx(kl[(B + 1) * BLOCO:T - 1].mean())


def test_mascara_sem_despejo_nao_muda_nada():
    modelo = qwen()
    with torch.no_grad():
        cheio = modelo(torch.tensor([IDS])).logits[0]
    assert torch.allclose(cheio, logits_com_mascara(modelo, np.tril(np.ones((1, NB, NB), dtype=bool))), atol=1e-5)


def test_mascara_por_cabeca_de_kv_vai_para_o_grupo_certo():
    """Por cabeça (sem equivalente físico com tensores retangulares): a cabeça de K/V 1 da
    camada 0 despeja o bloco 2; só as consultas 7..13 dessa camada param de vê-lo."""
    modelo = qwen(n_camadas=2)
    vis = np.tril(np.ones((2 * 2, NB, NB), dtype=bool))
    vis[1, B + 1:, 2] = False  # unidade 1 = camada 0, cabeça de K/V 1
    with Mascara(modelo, vis, BLOCO, T, ler_arquitetura(modelo)), torch.no_grad():
        att = modelo(torch.tensor([IDS]), output_attentions=True).attentions
    depois, bloco2 = slice((B + 1) * BLOCO, T), slice(2 * BLOCO, 3 * BLOCO)
    assert att[0][0, 7:, depois, bloco2].abs().max() == 0  # grupo 1: não vê mais
    assert att[0][0, :7, depois, bloco2].min() > 0  # grupo 0: continua vendo
    assert att[1][0, :, depois, bloco2].min() > 0  # outra camada: continua vendo

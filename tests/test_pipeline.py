"""Roda os scripts de ponta a ponta, com poucos dados e treino curto, nos três modos."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("matplotlib")

RAIZ = Path(__file__).resolve().parents[1]


def rodar(script: str, *args: str, cwd: Path):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(RAIZ / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMP_NUM_THREADS", "1")
    r = subprocess.run([sys.executable, str(RAIZ / "scripts" / script), *args],
                       cwd=cwd, env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"{script} falhou:\n{r.stdout}\n{r.stderr}"
    return r.stdout


def test_pipeline_completo(tmp_path):
    config = str(RAIZ / "configs" / "dqn.yml")
    rodar("gerar_traces_sinteticos.py", "--n-treino", "4", "--n-teste", "2", "--blocos", "24", cwd=tmp_path)
    for modo in ("geral", "por_camada", "por_cabeca"):
        rodar("avaliar_baselines.py", "--fonte", "sintetico", "--modo", modo, "--config", config, cwd=tmp_path)
        rodar("treinar_dqn.py", "--fonte", "sintetico", "--modo", modo, "--config", config,
              "--sementes", "0", "1", "--passos", "400", "--freq-avaliacao", "200", cwd=tmp_path)
        rodar("graficos.py", "--fonte", "sintetico", "--modo", modo, cwd=tmp_path)
        figuras = tmp_path / "resultados" / "sintetico" / modo / "figuras"
        assert (figuras / "curva_aprendizado.png").exists() and (figuras / "comparacao_final.png").exists()
    saida = rodar("comparar_modos.py", "--fonte", "sintetico", cwd=tmp_path)
    assert "Decisão por cabeça" in saida
    assert (tmp_path / "resultados" / "sintetico" / "comparacao_modos.png").exists()


def modelo_de_brinquedo(pasta: Path, texto: str):
    """Tokenizador BPE treinado na hora + Qwen2 com GQA de pesos aleatórios, salvos em disco."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator([texto], trainers.BpeTrainer(vocab_size=400, special_tokens=["<unk>", "<eos>"],
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", eos_token="<eos>")
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=56, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=14, num_key_value_heads=2, max_position_embeddings=1024,
                      eos_token_id=tok.eos_token_id, bos_token_id=tok.eos_token_id)
    Qwen2ForCausalLM(cfg).save_pretrained(pasta)
    tok.save_pretrained(pasta)


def test_pipeline_com_llm(tmp_path):
    """Grava traces de um LLM, treina, avalia no próprio LLM com o cache podado e faz os gráficos."""
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    import numpy as np

    rng = np.random.default_rng(0)
    palavras = "the model reads a long text and keeps only part of its memory while it writes".split()
    texto = " ".join(rng.choice(palavras, 12000))
    (tmp_path / "textos").mkdir()
    (tmp_path / "textos" / "a.txt").write_text(texto)
    modelo_de_brinquedo(tmp_path / "mini", texto)

    config = str(RAIZ / "configs" / "dqn.yml")
    rodar("gerar_traces_llm.py", "--modelo", "mini", "--fonte", "mini", "--textos", "textos",
          "--tokens", "320", "--n-treino", "3", "--n-teste", "2", cwd=tmp_path)
    for modo in ("geral", "por_camada", "por_cabeca"):
        rodar("treinar_dqn.py", "--fonte", "mini", "--modo", modo, "--config", config,
              "--sementes", "0", "--passos", "200", "--freq-avaliacao", "100", cwd=tmp_path)
        saida = rodar("avaliar_no_modelo.py", "--fonte", "mini", "--modo", modo, "--config", config, cwd=tmp_path)
        assert "dqn_seed0" in saida and "KL" in saida
        res = tmp_path / "resultados" / "mini" / modo
        assert (res / "modelo_real.csv").exists() and (res / "modelo_real_resumo.json").exists()
    rodar("graficos_modelo_real.py", "--fonte", "mini", cwd=tmp_path)
    for nome in ("comparacao_modos_real.png", "proxy_x_real.png"):
        assert (tmp_path / "resultados" / "mini" / nome).exists()

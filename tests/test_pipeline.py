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

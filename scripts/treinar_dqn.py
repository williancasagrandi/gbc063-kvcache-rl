"""Treina o DQN (Stable-Baselines3) no ambiente proxy, com várias sementes.

    python scripts/treinar_dqn.py --fonte gpt2 --modo geral
    python scripts/treinar_dqn.py --fonte gpt2 --modo grupos
    python scripts/treinar_dqn.py --fonte gpt2 --modo por_camada
    python scripts/treinar_dqn.py --fonte sintetico --modo geral --sementes 0 1 2

Hiperparâmetros e parâmetros do ambiente vêm de configs/dqn.yml (uma entrada por modo).
Treina nos traces de dados/<fonte>/treino e, a cada eval_freq passos, avalia a
política gulosa (sem exploração) em todos os traces de teste.

Saída em resultados/<fonte>/<modo>/dqn/seed<N>/:
    curva.csv     passos, epsilon, retorno, atenção perdida e agulha no teste
    final.json    avaliação do modelo final (= último ponto da curva)
    modelo.zip    rede treinada
    monitor.csv   retorno de cada episódio de treino (com exploração)
"""

import argparse
import csv
import json
import time
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from kvcache_rl import KVCacheEnv, carregar_pasta
from kvcache_rl.baselines import avaliar, politica_sb3
from kvcache_rl.config import CHAVES, carregar_config

CAMPOS = ["passos", "epsilon", "retorno", "retorno_dp", "atencao_perdida", "agulha"]


class AvaliacaoPeriodica(BaseCallback):
    """Avalia a política gulosa nos traces de teste a cada `freq` passos."""

    def __init__(self, env_teste, freq: int, total: int, caminho_csv: Path):
        super().__init__()
        self.env_teste, self.freq, self.total, self.caminho = env_teste, freq, total, caminho_csv
        self.linhas = []
        self.final = None

    def _avaliar(self):
        r = avaliar(self.env_teste, politica_sb3(self.model))
        eps = self.model.exploration_rate if self.num_timesteps > 0 else self.model.exploration_initial_eps
        linha = {"passos": self.num_timesteps, "epsilon": eps, **r}
        self.linhas.append(linha)
        with open(self.caminho, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CAMPOS)
            w.writeheader()
            w.writerows(self.linhas)
        print(f"  passo {self.num_timesteps:>7}  eps {linha['epsilon']:.2f}  "
              f"retorno {r['retorno']:7.2f}  perdida {r['atencao_perdida']:.3f}  agulha {r['agulha']:.2f}",
              flush=True)
        return r

    def _on_training_start(self):
        self._avaliar()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.freq == 0 and self.num_timesteps < self.total:
            self._avaliar()
        return True

    def _on_training_end(self):
        # o último ponto da curva e o final.json são a mesma avaliação, do modelo já salvo
        self.final = self._avaliar()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fonte", default="gpt2", help="gpt2 ou sintetico (pasta em dados/)")
    p.add_argument("--modo", default="geral", choices=list(CHAVES))
    p.add_argument("--config", default="configs/dqn.yml")
    p.add_argument("--sementes", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--passos", type=int, default=None, help="sobrescreve n_timesteps do yml")
    p.add_argument("--freq-avaliacao", type=int, default=None, help="sobrescreve eval_freq do yml")
    a = p.parse_args()

    cfg = carregar_config(a.config, a.modo)
    n_passos = a.passos or cfg["passos"]
    freq = a.freq_avaliacao or cfg["eval_freq"]

    treino = carregar_pasta(f"dados/{a.fonte}/treino")
    teste = carregar_pasta(f"dados/{a.fonte}/teste")
    env_treino = KVCacheEnv(treino, **cfg["env"])
    env_teste = KVCacheEnv(teste, **cfg["env"])
    for tr in treino + teste:
        tr.camadas = None  # o ambiente já guardou o que precisa; libera memória
    print(f"modo {a.modo}: {len(treino)} traces de treino, {len(teste)} de teste ({a.fonte}); "
          f"{env_teste.decisoes_por_texto} decisões por texto; {n_passos} passos por semente")

    for seed in a.sementes:
        pasta = Path(f"resultados/{a.fonte}/{a.modo}/dqn/seed{seed}")
        pasta.mkdir(parents=True, exist_ok=True)
        print(f"\n== {a.modo}, semente {seed} ==")

        env = Monitor(env_treino, str(pasta / "monitor.csv"))
        modelo = DQN(cfg["policy"], env, seed=seed, device="cpu", verbose=0, **cfg["dqn"])

        inicio = time.time()
        cb = AvaliacaoPeriodica(env_teste, freq, n_passos, pasta / "curva.csv")
        modelo.learn(total_timesteps=n_passos, callback=cb)
        modelo.save(pasta / "modelo")

        final = dict(cb.final)
        final.update(semente=seed, passos=n_passos, minutos=round((time.time() - inicio) / 60, 1))
        (pasta / "final.json").write_text(json.dumps(final, indent=2))
        print(f"final: {final}")


if __name__ == "__main__":
    main()

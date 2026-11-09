# Uso:  make tudo                 (GPT-2, recomendado)
#       make tudo FONTE=sintetico (sem GPU e sem internet)
FONTE ?= gpt2
PY ?= .venv/bin/python

.PHONY: setup setup-gpt2 teste traces baselines treinar graficos tudo limpar

setup:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

setup-gpt2: setup
	$(PY) -m pip install -r requirements-gpt2.txt

teste:
	$(PY) -m pytest -q

traces:
ifeq ($(FONTE),gpt2)
	$(PY) scripts/gerar_traces_gpt2.py
else
	$(PY) scripts/gerar_traces_sinteticos.py
endif

baselines:
	$(PY) scripts/avaliar_baselines.py --fonte $(FONTE)

treinar:
	$(PY) scripts/treinar_dqn.py --fonte $(FONTE)

graficos:
	$(PY) scripts/graficos.py --fonte $(FONTE)

tudo: traces baselines treinar graficos

limpar:
	rm -rf dados/$(FONTE) resultados/$(FONTE)

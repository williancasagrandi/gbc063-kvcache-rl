# Uso:
#   make tudo                       um modo (MODO=geral por padrão), GPT-2
#   make tudo MODO=por_camada       outro modo
#   make tudo-modos                 os três modos + gráfico comparando (~40 min)
#   make tudo FONTE=sintetico       sem GPU e sem internet
FONTE ?= gpt2
MODO ?= geral
MODOS = geral grupos por_camada
PY ?= .venv/bin/python

.PHONY: setup setup-gpt2 teste traces baselines treinar graficos tudo modos comparar tudo-modos limpar

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
	$(PY) scripts/avaliar_baselines.py --fonte $(FONTE) --modo $(MODO)

treinar:
	$(PY) scripts/treinar_dqn.py --fonte $(FONTE) --modo $(MODO)

graficos:
	$(PY) scripts/graficos.py --fonte $(FONTE) --modo $(MODO)

tudo: traces baselines treinar graficos

# os três modos sobre os mesmos traces, depois o gráfico comparando
modos:
	for m in $(MODOS); do \
		$(PY) scripts/avaliar_baselines.py --fonte $(FONTE) --modo $$m && \
		$(PY) scripts/treinar_dqn.py --fonte $(FONTE) --modo $$m && \
		$(PY) scripts/graficos.py --fonte $(FONTE) --modo $$m || exit 1; \
	done
	$(PY) scripts/comparar_modos.py --fonte $(FONTE)

comparar:
	$(PY) scripts/comparar_modos.py --fonte $(FONTE)

tudo-modos: traces modos

limpar:
	rm -rf dados/$(FONTE) resultados/$(FONTE)

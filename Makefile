# Uso:
#   make tudo                       um modo (MODO=geral por padrão), GPT-2
#   make tudo MODO=por_cabeca       outro modo (geral, por_camada, por_cabeca)
#   make tudo-modos                 os três modos + gráfico comparando (~1 h)
#   make tudo-modos FONTE=qwen      o mesmo com o Qwen2.5-0.5B (GQA: 48 caches)
#   make tudo FONTE=sintetico       sem GPU e sem internet
#   make real MODO=geral            só a avaliação no LLM real (precisa dos traces e do DQN)
# Variáveis: FONTE (gpt2|qwen|sintetico), MODO, TOKENS (tamanho dos textos), PY
#
# Com FONTE gpt2 ou qwen, as políticas também são avaliadas no próprio LLM com o cache
# podado (perplexidade e senha): são essas as métricas finais. Os traces sintéticos não
# têm um modelo por trás, então para eles só existem as métricas do proxy.
FONTE ?= gpt2
MODO ?= geral
MODOS = geral por_camada por_cabeca
TOKENS ?=
PY ?= .venv/bin/python
ARGS_TRACES = $(if $(TOKENS),--tokens $(TOKENS),)

.PHONY: setup setup-llm setup-gpt2 teste traces baselines treinar graficos real tudo modos comparar tudo-modos limpar

setup:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

# transformers + datasets, para gravar traces de um LLM real (GPT-2 ou Qwen)
setup-llm: setup
	$(PY) -m pip install -r requirements-llm.txt

setup-gpt2: setup-llm

teste:
	$(PY) -m pytest -q

traces:
ifeq ($(FONTE),sintetico)
	$(PY) scripts/gerar_traces_sinteticos.py
else
	$(PY) scripts/gerar_traces_llm.py --fonte $(FONTE) $(ARGS_TRACES)
endif

baselines:
	$(PY) scripts/avaliar_baselines.py --fonte $(FONTE) --modo $(MODO)

treinar:
	$(PY) scripts/treinar_dqn.py --fonte $(FONTE) --modo $(MODO)

graficos:
	$(PY) scripts/graficos.py --fonte $(FONTE) --modo $(MODO)

real:
	$(PY) scripts/avaliar_no_modelo.py --fonte $(FONTE) --modo $(MODO)

ifeq ($(FONTE),sintetico)
tudo: traces baselines treinar graficos
else
tudo: traces baselines treinar graficos real
endif

# os três modos sobre os mesmos traces, depois os gráficos comparando
modos:
	for m in $(MODOS); do \
		$(PY) scripts/avaliar_baselines.py --fonte $(FONTE) --modo $$m && \
		$(PY) scripts/treinar_dqn.py --fonte $(FONTE) --modo $$m && \
		$(PY) scripts/graficos.py --fonte $(FONTE) --modo $$m || exit 1; \
		if [ "$(FONTE)" != "sintetico" ]; then \
			$(PY) scripts/avaliar_no_modelo.py --fonte $(FONTE) --modo $$m || exit 1; \
		fi; \
	done
	$(MAKE) comparar FONTE=$(FONTE)

comparar:
	$(PY) scripts/comparar_modos.py --fonte $(FONTE)
	if [ "$(FONTE)" != "sintetico" ]; then $(PY) scripts/graficos_modelo_real.py --fonte $(FONTE); fi

tudo-modos: traces modos

limpar:
	rm -rf dados/$(FONTE) resultados/$(FONTE)

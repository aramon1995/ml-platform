DC = docker compose
RUN = $(DC) exec prefect-serve prefect deployment run --watch
MODELS ?=
FEATURES ?=
TRIALS ?=
INPUT ?=
ROWS ?=

comma := ,
train_params = $(if $(MODELS),--param 'models=["$(subst $(comma),"$(comma)",$(MODELS))"]') \
	$(if $(FEATURES),--param feature_set=$(FEATURES)) \
	$(if $(TRIALS),--param n_trials=$(TRIALS))

.PHONY: start stop clean seed train select train-select infer eda test

start:
	$(DC) up -d --build --wait --remove-orphans
	$(DC) run --rm seed

stop:
	$(DC) down

clean:
	$(DC) down -v

seed:
	$(DC) run --rm seed

train:
	$(RUN) train/train $(train_params)

select:
	$(RUN) select/select

train-select:
	$(RUN) train-and-select/train-and-select $(train_params)

infer:
	$(RUN) batch-inference/batch-inference $(if $(INPUT),--param input_uri=$(INPUT)) $(if $(ROWS),--param 'rows=$(ROWS)')

eda:
	uv run jupyter nbconvert --to notebook --execute --inplace lab/eda.ipynb

test:
	uv run pytest

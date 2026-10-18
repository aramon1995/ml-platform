DC = docker compose
RUN = $(DC) exec prefect-serve prefect deployment run --watch
MODELS ?=
TRIALS ?=
INPUT ?=

comma := ,
models_json = $(if $(MODELS),"models": ["$(subst $(comma),"$(comma)",$(MODELS))"],)
trials_json = $(if $(TRIALS),"n_trials": $(TRIALS),)
train_params = {$(models_json)$(if $(and $(MODELS),$(TRIALS)),$(comma))$(trials_json)}

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
	$(RUN) train/train --params '$(train_params)'

select:
	$(RUN) select/select

train-select:
	$(RUN) train-and-select/train-and-select --params '$(train_params)'

infer:
	$(RUN) batch-inference/batch-inference $(if $(INPUT),--params '{"input_uri": "$(INPUT)"}')

eda:
	uv run jupyter nbconvert --to notebook --execute --inplace lab/eda.ipynb

test:
	uv run pytest

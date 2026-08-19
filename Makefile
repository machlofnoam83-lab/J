.PHONY: install download prepare train evaluate chat serve test

install:
	python -m pip install -e ".[dev]"
	jai-download

download:
	jai-download

prepare:
	jai-prepare data/sample_train.jsonl data/sample_eval.jsonl

train:
	jai-train --config configs/train_4060.yaml

evaluate:
	jai-evaluate --model outputs/jai-3b-he

chat:
	jai-chat --model outputs/jai-3b-he

serve:
	jai-serve --model outputs/jai-3b-he --host 0.0.0.0

test:
	pytest -q

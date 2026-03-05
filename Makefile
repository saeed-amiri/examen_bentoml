# ===============================
# Makefile for examen_bentoml
# ===============================

PYTHON = python3.12
VENV = .venv
PIP = $(VENV)/bin/python -m pip
PY = $(VENV)/bin/python
PYTEST = $(VENV)/bin/python -m pytest
BENTOML = $(VENV)/bin/python -m bentoml

# ------------------
# Create virtual environment and install dependencies
# ------------------
setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Environment installed in $(VENV)"

# ------------------
# Prepare dataset
# ------------------
prepare:
	$(PY) src/prepare_data.py

# ------------------
# Run Docker container
# ------------------
run:
	docker run --rm -p 3000:3000 admission_service:1.0.0

# ------------------
# Full pipeline: prepare data and train model
# ------------------
all: prepare train
	@echo "Data prepared and model trained"

# ------------------
# Train model
# ------------------
train:
	$(PY) src/train_model.py

# ------------------
# Serve BentoML service
# ------------------
serve:
	$(BENTOML) serve src.service:admission_service --reload

# ------------------
# Run tests
# ------------------
test:
	PYTHONPATH=./src $(PYTEST) -v

# ------------------
# Build BentoML container
# ------------------
build:
	$(BENTOML) build --version 1.0.0
	$(BENTOML) containerize admission_service:1.0.0

save:
	docker save -o admission_service.tar admission_service:1.0.0

restore:
	docker load -i admission_service.tar
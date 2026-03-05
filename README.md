# Admissions Prediction — BentoML (Exam)

Prediction API service for student admission probability using LinearRegression + StandardScaler.  
Packaged with **BentoML 1.4.x** and **Docker**.  
Authentication via **JWT** (`/login`) + secure prediction endpoint `/v1/models/admission_lr/predict`.

---

## Repository structure (exam)

```text

examen_bentoml/
├── data/
│   ├── raw/               # admission.csv (source)
│   └── processed/         # X_train.csv, y_train.csv, X_test.csv, y_test.csv
├── src/
│   ├── prepare_data.py    # load + clean + normalize columns (snake_case)
│   ├── train_model.py     # sklearn pipeline (StandardScaler + LinearRegression) + save BentoML
│   └── service.py         # BentoML service (JWT + endpoints)
├── tests/
│   └── test_endpoints.py  # unit tests (pytest)
├── bentofile.yaml         # BentoML config (bento build)
├── admission_service.tar  # docker image archive of the bento
├── requirements.txt       # development dependencies
├── Makefile               # shortcut commands
└── README.md              # this file

````

---

## Prerequisites

- **Docker** (required for evaluation)  
- **Python 3.12** (only if running local development)  
- Port **3000** available  

---

## Evaluation workflow (recommended)

> The reviewer will have a **Docker image** (`admission_service.tar`) and the structure to run tests:  
> - `tests/test_endpoints.py`  
> - `Makefile`  
> - `requirements.txt`  
>
> Goal: **load the image, run the API, run tests**.

1. **Extract the archive** containing the submission files + Makefile/requirements.txt.

2. **Load the Docker image**:
```bash
make restore
# equivalent: docker load -i admission_service.tar
````

3. **Run the API**:

```bash
make run
# equivalent: docker run --rm -p 3000:3000 admission_service:1.0.0
# API listens on http://127.0.0.1:3000
```

4. **Setup local test environment** (pytest + requests):

```bash
make setup
# equivalent: python3 -m venv .venv && pip install --upgrade pip && pip install -r requirements.txt
```

5. **Run unit tests**:

```bash
pytest -v
# optional via Makefile: make test
# all tests should pass
```

> Note: tests call the running API over HTTP, so start the container (`make run`) before running tests.

> The **Make targets** exist to make the validation process easier.
> You can run the raw commands if you prefer.

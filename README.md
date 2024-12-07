# MTGA-CV-Voice-Interface

## Project Setup

Activate venv:
```bash
.venv\Scripts\activate
```

From active venv:
```bash
ipython kernel install --user --name=.venv
```

Create/open an ipynb file and select .venv as the kernel.

## Labeling and Auto-Annotation

Clone and install label studio backend from repository
```bash
git clone https://github.com/HumanSignal/label-studio-ml-backend.git
cd label-studio-ml-backend/
pip install -e .
```

If you need to create the backend
```bash
label-studio-ml create my_ml_backend
```
- Replace the content of my_ml_backend/model.py with ml_backend_model.py

From the label-studio-ml-backend repo, run this. It should pass the health check:
```bash
label-studio-ml start .\my_ml_backend
```

To start the frontend, run `label-studio` in the terminal
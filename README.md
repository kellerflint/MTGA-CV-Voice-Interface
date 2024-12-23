# MTGA-CV-Voice-Interface

## Project Overview

This application allows users to play Magic the Gathering Arena (MTGA) through the use of voice commands. 

The repository includes:
- A demo of the application (application_demo.ipynb)
- Scripts for data collection (data_collection.ipynb)
- Exploration of different approaches (model_exploration.ipynb)
- Model Training and Validation (model_train.ipynb)
- An endpoint for Label Studio's API to perform automated first pass annotations (my_ml_backend_model.py)

The application itself works as follows:
- Listens for user input over audio device
- Transcribes audio when it detects the end of voice activity
- Takes a screenshot of the main window
- Makes use of the tuned YOLO model to detect classes and bounding boxes for cards and buttons
- Uses OCR to extract the text on the objects to identify them
- Calls an LLM that maps the user command to the extracted card text
- LLM returns tool calls to move cursor, click objects, and play cards as directed by the user

## Project Setup

### Activate Virtual Environment

Create venv
```bash
python -m venv .venv
```

Activate venv:
```bash
.venv\Scripts\activate
```

Install from requirements.txt
```bash
pip install -r requirements.txt
```

From active venv:
```bash
ipython kernel install --user --name=.venv
```

Create/open an ipynb file and select .venv as the kernel (under the Jupyter options).

### Labeling and Auto-Annotation

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
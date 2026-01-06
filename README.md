# Airflights — Flight Delay Prediction & Analysis

A compact, well-organized Python project that implements an end-to-end machine learning pipeline for predicting and analyzing flight delays. It contains data ingestion and preprocessing, model training and evaluation, inference utilities, and a Streamlit demo application for interactive exploration.

## Key Features
- Reproducible data preprocessing pipeline
- Model training and evaluation with saved artifacts
- Inference utilities for single and batch predictions
- Interactive Streamlit app for visualization and demonstration

## Installation

Prerequisites: Python 3.8+ recommended. Create and use a virtual environment for isolation.

1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

If you only plan to run the Streamlit demo and it isn't listed, install Streamlit explicitly:

```powershell
pip install streamlit
```

## Usage

- Run the Streamlit demo UI:

```powershell
streamlit run app/streamlit_app.py
```

- Train a model (example):

```powershell
python src/train.py
```

- Run inference (example):

```powershell
python src/inference.py
```

Check `src/train.py` and `src/inference.py` for available command-line arguments and configuration options.

## Project Structure

```
Project root
├─ app/
│  └─ streamlit_app.py
├─ assets/
├─ data/
│  ├─ raw/
│  └─ preprocessed/
├─ models/
│  └─ saved_models/
├─ NoteBooks/
│  └─ End-to-End Data Science Project.ipynb
├─ src/
│  ├─ __init__.py
│  ├─ data_pipeline.py
│  ├─ evaluation.py
│  ├─ inference.py
│  ├─ model.py
│  ├─ train.py
│  └─ utils.py
├─ requirements.txt
└─ README.md
```

- `app/`: Streamlit application and demo assets
- `data/raw/`: Original datasets (not committed unless necessary)
- `data/preprocessed/`: Cleaned and feature-engineered datasets used for training
- `models/saved_models/`: Trained model artifacts and serializers
- `src/`: Core pipeline and model code
- `NoteBooks/`: Exploratory analysis and experiments

## Dependencies

All required Python packages and pinned versions should be listed in `requirements.txt`. Typical dependencies include:

- `pandas`, `numpy` — data handling
- `scikit-learn` — modeling and evaluation
- `joblib` or `pickle` — model serialization
- `streamlit` — web demo
- `matplotlib`, `seaborn` — visualization

Install with:

```powershell
pip install -r requirements.txt
```

## License

This repository is distributed under the MIT License by default. To change the license, replace this section and add a `LICENSE` file.

```text
MIT License

Copyright (c) YEAR Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

[Replace YEAR and Your Name with appropriate values.]
```

## Contributing

Contributions are welcome. Please open an issue to discuss changes before submitting a pull request. Provide clear descriptions and, where appropriate, update tests and documentation.

## Contact

### 🧑‍💻 **Abdallah Nabil Ragab**  
**Data Scientist | Machine Learning Engineer | Software Engineer**  
**M.Sc. in Business Information Systems**

If you have any suggestions, ideas, feature requests, or want to report issues,  
please feel free to send your feedback directly via email:

📩 **Email:** `abdallah.nabil.ragab94@gmail.com`  

I appreciate your thoughts and feedback that help improve this project.


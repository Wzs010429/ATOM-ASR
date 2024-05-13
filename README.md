# ATOM-ASR

File structure:

```
ASR_Project/
│
├── data/
│   ├── raw/                  # Store raw data, possibly large audio files
│   ├── processed/            # Transformed data ready for modeling
│   └── external/             # Data from third-party sources
│
├── notebooks/                # Jupyter notebooks for exploration and presentation
│   ├── EDA.ipynb             # Exploratory Data Analysis
│   ├── Model_Training.ipynb  # Notebooks for training models
│   └── Evaluation.ipynb      # Comparison and evaluation of models
│
├── src/                      # Source code for use in this project
│   ├── __init__.py           # Makes src a Python module
│   ├── data/                 # Scripts to download or generate data
│   ├── features/             # Scripts to turn raw data into features for modeling
│   ├── models/               # Scripts to train models and then use trained models to make predictions
│   │   ├── train_model.py    # Script to train and save models
│   │   └── predict_model.py  # Script to load models and make predictions
│   └── visualization/        # Scripts to create exploratory and results oriented visualizations
│
├── dashboards/               # Directory to store demo dashboard files
│   ├── app.py                # Main file to run the dashboard
│   └── components/           # Reusable components for the dashboard
│       └── __init__.py       # Makes components a Python package
│
├── models/                   # Trained and serialized models, model predictions, or model summaries
│
├── reports/                  # Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── figures/              # Generated graphics and figures to be used in reporting
│   └── performance_reports/  # Performance reports of models
│
├── requirements.txt          # The dependencies file for reproducing the analysis environment
├── Dockerfile                # Dockerfile for containerizing the environment
└── README.md                 # The top-level README for developers using this project
```
import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report

# Config
GPT_MODEL = os.environ.get('GPT_MODEL', 'gpt2')
EPOCHS = int(os.environ.get('EPOCHS', 10))
EMOTIONS = ['joy', 'love', 'fear', 'sadness', 'anger', 'surprise']
GENRE = os.environ.get('GENRE', '')

CSV_RESULTS_DIR = 'results/emotion_labels.csv'
if GENRE:
    PATHS = [f'../models/{GPT_MODEL}/{emotion}/{GENRE}/{EPOCHS}_epochs/{CSV_RESULTS_DIR}' for emotion in EMOTIONS]
    SAVE_RESULTS_DIR = os.environ.get('SAVE_RESULTS_DIR', f'../models/{GPT_MODEL}/{EPOCHS}_epochs/results/{GENRE}')
else:
    PATHS = [f'../models/{GPT_MODEL}/{emotion}/{EPOCHS}_epochs/{CSV_RESULTS_DIR}' for emotion in EMOTIONS]
    SAVE_RESULTS_DIR = os.environ.get('SAVE_RESULTS_DIR', f'../models/{GPT_MODEL}/{EPOCHS}_epochs/results')

def evaluate(filepaths):
    
    df = pd.concat(map(pd.read_csv, filepaths))
    df = df.astype({'target_labels': 'str',
                    'target_confidence': 'float',
                    'predicted_labels': 'str',
                    'predicted_confidence': 'float'
                    })
    df = df.dropna()

    y_true = df['target_labels'].to_numpy()
    y_pred = df['predicted_labels'].to_numpy()
    
    report = classification_report(y_true, y_pred, labels=EMOTIONS, target_names=EMOTIONS)
    mean_averages = df.groupby('target_labels')['target_confidence'].mean()
    
    return report, mean_averages, y_pred

if __name__=='__main__':
    print("Evaluating...")
    report, _, _ = evaluate(filepaths=PATHS)
    formatted_results = f"RESULTS\n{report}"

    if not os.path.exists(SAVE_RESULTS_DIR):
        os.makedirs(SAVE_RESULTS_DIR)

    with open(SAVE_RESULTS_DIR + "/results.txt", "w") as f:
        f.write(formatted_results)

    print("Done.")
    
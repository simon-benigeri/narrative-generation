import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForSeq2SeqLM

EMOTION_SCORE_THRESHOLD = 4
EMOTIONS = ['joy', 'love', 'fear', 'sadness', 'anger', 'surprise']

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")
# model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-emotion")
model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-emotion")


def evaluate(samples):
    outputs = np.array([list(_get_emotion(sample)) for sample in samples])
    df = pd.DataFrame(data=outputs,
                      columns = ['target_labels', 'target_confidence', 'predicted_labels', 'predicted_confidence'])
    df = df.astype({'target_labels': 'str',
                    'target_confidence': 'float',
                    'predicted_labels': 'str',
                    'predicted_confidence': 'float'
                    })
    y_true = df['target_labels'].to_numpy()
    y_pred = df['predicted_labels'].to_numpy()

    report = classification_report(y_true, y_pred, labels=EMOTIONS, target_names=EMOTIONS)
    mean_averages = df.groupby('target_labels')['target_confidence'].mean()
    return report, mean_averages

def _get_emotion(text):
    texts = text.split('\n')[1].split(':')
    target, response = texts[1].strip(' ()'), texts[2].strip()

    input_ids = tokenizer.encode(response + '</s>', return_tensors='pt')
    output = model.generate(input_ids=input_ids, max_length=2, return_dict_in_generate=True,  output_scores=True)

    # Get emotion label scores
    scores = [output.scores[0][0][tokenizer.encode(emotion)[0]].item() for emotion in EMOTIONS]
    scores = list(map(float, list(torch.nn.functional.softmax(torch.tensor(scores), dim=0).detach().numpy())))

    predicted = EMOTIONS[scores.index(max(scores))]
    target_confidence = scores[EMOTIONS.index(target)]
    predicted_confidence = max(scores)

    return (target, target_confidence, predicted, predicted_confidence)

if __name__=='__main__':
    s = "C: (joy): What did they say?\nR: (sadness): They said I'm a worthless human being."
    a = "C: (joy): What did they say?\nR: (anger): They said I'm fired."
    e = "C: (joy): What did they say?\nR: (anger): They said I don't have a reservation."
    f = "C: (joy): What did they say?\nR: (anger): They said they love me."
    g = "C: (joy): What did they say?\nR: (anger): They said I'm gonna be rich."
    test = [s, a, e, f, g]
    out = evaluate(test)
    print(out)

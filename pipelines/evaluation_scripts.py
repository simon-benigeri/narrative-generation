import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForSeq2SeqLM

EMOTION_SCORE_THRESHOLD = 4
EMOTIONS = ['joy', 'love', 'fear', 'sadness', 'anger', 'surprise']


def evaluate(model, samples):
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

def _get_emotion(model, text):
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

def generate_samples(model, prompt):
    model.eval()
    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(device)

    sample_outputs = model.generate(
                                generated, 
                                #bos_token_id=random.randint(1,30000),
                                do_sample=True,   
                                top_k=50, 
                                max_length = 300,
                                top_p=0.90, 
                                num_return_sequences=3
                            )

    return tokenizer.decode(sample_output, skip_special_tokens=True))

if __name__=='__main__':
    # Load emotion model
    emotion_tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")
    # emotion_model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-emotion")
    emotion_model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-emotion")

    # Load dialogue model
    # ...

    #generated = generate_samples(dialogue_model, prompt)
    out = evaluate(emotion_model, generated)
    print(out)

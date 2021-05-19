import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForSeq2SeqLM, GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

EMOTION_SCORE_THRESHOLD = 4
EMOTIONS = ['joy', 'love', 'fear', 'sadness', 'anger', 'surprise']
LOAD_DIALOGUE_MODEL_DIR = os.environ.get('LOAD_DIALOGUE_MODEL_DIR', 'models/temp')
NUM_GENERATIONS = int(os.environ.get('NUM_GENERATIONS', 3))
NUM_SAMPLES = int(os.environ.get('NUM_SAMPLES', 1))

def evaluate(model, samples):
    outputs = np.array([list(_get_emotion(model, sample)) for sample in samples])
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

def generate_samples(model, prompt, num_samples=1):
    model.eval()
    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(model.device)

    sample_outputs = model.generate(
                                generated, 
                                #bos_token_id=random.randint(1,30000),
                                do_sample=True,   
                                top_k=50, 
                                max_length = 300,
                                top_p=0.90, 
                                num_return_sequences=num_samples
                            )

    return [tokenizer.decode(sample_output, skip_special_tokens=True) for sample_output in sample_outputs]

if __name__=='__main__':
    print("Loading models...")
    # Load emotion model
    emotion_tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")
    # emotion_model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-emotion")
    emotion_model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-emotion")

    # Load dialogue model
    tokenizer = GPT2Tokenizer.from_pretrained(LOAD_DIALOGUE_MODEL_DIR, bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
    configuration = GPT2Config.from_pretrained(LOAD_DIALOGUE_MODEL_DIR, output_hidden_states=False)
    dialogue_model = GPT2LMHeadModel.from_pretrained(LOAD_DIALOGUE_MODEL_DIR, config=configuration)
    dialogue_model.resize_token_embeddings(len(tokenizer))
    dialogue_model.eval()

    print("Evaluating...")
    generated = generate_samples(dialogue_model, "C: (neutral): Yo jim you got a minute?\nR: (fear):", num_samples=NUM_SAMPLES)
    
    for g in generated:
        print(g)
        print()

    report, mean_averages = evaluate(emotion_model, generated)
    print(report)

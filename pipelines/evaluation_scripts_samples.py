import os
import numpy as np
import pandas as pd
import re
import os
import random
from typing import List
import torch
import json
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForSeq2SeqLM, GPT2Tokenizer, GPT2Config, \
    GPT2LMHeadModel
from utils import *

SCRIPTS_DIR = os.environ.get('LOAD_SCRIPT_DATA_DIR', '../data/processed/formatted/call_responses/')
LOAD_DIALOGUE_MODEL_DIR = os.environ.get('LOAD_DIALOGUE_MODEL_DIR', 'models/temp/model_save')
SAVE_RESULTS_DIR = os.environ.get('SAVE_RESULTS_DIR', 'models/temp/results_samples')
THRESHOLD = float(os.environ.get('THRESHOLD', 0.7))
TARGET_EMOTION = os.environ.get('TARGET_EMOTION', "")
NUM_SAMPLES = int(os.environ.get('NUM_SAMPLES', 1))
NUM_PROMPTS = int(os.environ.get('NUM_PROMPTS', 10))

SAVE_PROMPTS = bool(os.environ.get('SAVE_PROMPTS', True))
LOAD_PROMPTS = bool(os.environ.get('LOAD_PROMPTS', False))
TEST_SET_DIR = os.environ.get('TEST_SET_DIR', '../models/temp/test_sets')


SAMPLE_PROMPTS = [
      "C: Hey, what happened?\nR: ",
      "C: What did they say?\nR: ",
      "C: So, how did it go?\nR: ",
      "C: How do you feel?\nR: ",
      "C: I guess we have to.\nR: ",
      "C: It's over.\nR: ",
      "C: I have to go find him now.\nR: ",
      "C: The party was yesterday.\nR: ",
      "C: How are we going to break the news to him?\nR: ",
      "C: Wow! He actually won.\nR: ",
      "C: We have to leave.\nR: ",
      "C: You have to stop doing that.\nR: ",
      "C: Explain what happened.\nR: ",
      "C: Why did you do this?\nR: ",
      "C: How do you think it's going to go?\nR: ",
    ]

# Get all config values and hyperparameters
with open("config.yml", "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)

""" Evaluate how well the predicted emotion label with the original emotion label """
def evaluate(model, tokenizer, samples, emotion_tags_included=True, target_response_emotion=""):
    outputs = np.array([list(_get_emotion(model, tokenizer, sample)) for sample in samples])
    df = pd.DataFrame(data=outputs,
                      columns=['target_labels', 'target_confidence', 'predicted_labels', 'predicted_confidence'])
    df = df.astype({'target_labels': 'str',
                    'target_confidence': 'float',
                    'predicted_labels': 'str',
                    'predicted_confidence': 'float'
                    })
    # errors_df = df[df[predicted_labels].isna()]
    df = df.dropna()

    # Save table to use for combined table
    if not os.path.exists(SAVE_RESULTS_DIR):
        os.makedirs(SAVE_RESULTS_DIR)

    df.to_csv(SAVE_RESULTS_DIR + '/emotion_labels.csv', index=False)

    y_true = df['target_labels'].to_numpy()
    y_pred = df['predicted_labels'].to_numpy()

    report = classification_report(y_true, y_pred, labels=config.EMOTIONS, target_names=config.EMOTIONS)
    mean_averages = df.groupby('target_labels')['target_confidence'].mean()

    return report, mean_averages, y_pred

""" Get predicted and actual score of response along with respective confidence of each """
def _get_emotion(model, tokenizer, call_response):
    target = call_response["response"]["emotion"]
    response = call_response["response"]["text"]

    try:
        scores = get_emotion_scores(model, tokenizer, response)

        predicted = config["EMOTIONS"][scores.index(max(scores))] if max(scores) > THRESHOLD else "neutral"
        target_confidence = scores[config["EMOTIONS"].index(target)]
        predicted_confidence = max(scores) if max(scores) > THRESHOLD else 1 - max(scores)

    except Exception as e:
        print('Error with : ', response, " : ", e)
        predicted = None
        target_confidence = None
        predicted_confidence = None

    return (target, target_confidence, predicted, predicted_confidence)

""" Convert call-response text into dict """
def get_call_response_dict(generated, target_emotion="", allow_empty=False):
    generated_call_responses = []
    generated_original = []
    for g in generated:
        generated_call = g.split("\n")[0]
        generated_response = g.split("\n")[1]

        if target_emotion == "":
            generated_call_response = {
                "call": {"emotion": generated_call.split(":")[1].strip(" ()"),
                         "text": generated_call.split(":")[-1].strip()},
                "response": {"emotion": generated_response.split(":")[1].strip(" ()"),
                             "text": generated_response.split(":")[-1].strip()}
            }
        else:
            generated_call_response = {
                "call": {"emotion": "neutral", "text": generated_call.split(":")[-1].strip()},
                "response": {"emotion": target_emotion, "text": generated_response.split(":")[-1].strip()}
            }

        # Add if not empty
        if generated_call_response["response"]["text"] != "" or allow_empty:
            generated_call_responses.append(generated_call_response)
            generated_original.append(g)

    return generated_call_responses, generated_original

""" Get list of formatted call/responses for evaluation """
def get_formatted_call_responses(scripts_dir:str = SCRIPTS_DIR, num_scripts=0):
    lines = []
    for genre in os.listdir(scripts_dir):
        genre_emotion_dir = os.path.join(scripts_dir, genre, 'emotions')

        for i, filename in enumerate(os.listdir(genre_emotion_dir)):
            if num_scripts > 0 and i >= num_scripts:
                break

            fpath = os.path.join(genre_emotion_dir, filename)

            if os.path.splitext(fpath)[1] == '.txt':
                with open(fpath) as f:
                    lines += "".join(f.readlines()).split("---")

    return lines

""" Randomly select calls from call_responses as prompts """
def sample_prompts(call_responses:List[str], num_prompts:int = NUM_PROMPTS,threshold:int = 15, response_emotion:str = 'neutral'):
    prompts = []

    for text in call_responses:
        text = text.strip()
        try:
            call = text.split("\n")[0]
            response = text.split("\n")[1]
            call_response = {
                "call": {
                    "emotion": call.split(":")[1].strip(" ()"),
                    "text": call.split(":")[-1].strip()
                },
                "response": {
                    "emotion": response.split(":")[1].strip(" ()"),
                    "text": response.split(":")[-1].strip()
                }
            }
            if call_response["response"]["emotion"] == response_emotion and len(
                    call_response["response"]["text"]) > threshold:
                prompt = f'C: {call_response["call"]["text"]}\nR: '
                prompts.append(prompt)
        except:
            pass

    return random.sample(prompts, num_prompts)


if __name__ == '__main__':
    print("Loading models...")
    print("Loading T5 emotion tagger...")
    # Load emotion model
    emotion_tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")
    emotion_model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-emotion")
    emotion_model.resize_token_embeddings(len(emotion_tokenizer))
    emotion_model.eval()

    # Load dialogue model
    print("Loading GPT2 dialogue...")
    dialogue_tokenizer = GPT2Tokenizer.from_pretrained(LOAD_DIALOGUE_MODEL_DIR, bos_token='<|startoftext|>',
                                                       eos_token='<|endoftext|>', pad_token='<|pad|>')
    configuration = GPT2Config.from_pretrained(LOAD_DIALOGUE_MODEL_DIR, output_hidden_states=False)
    dialogue_model = GPT2LMHeadModel.from_pretrained(LOAD_DIALOGUE_MODEL_DIR, config=configuration)
    dialogue_model.resize_token_embeddings(len(dialogue_tokenizer))
    dialogue_model.eval()

    print("Generating...")

    if LOAD_PROMPTS:
        print("Loading prompts...")
        with open(TEST_SET_DIR + f"/test_set__{NUM_PROMPTS}_prompts.json") as f:
            prompts = json.load(f)['PROMPTS']
    else:
        print("Sampling prompts...")
        call_responses = get_formatted_call_responses(scripts_dir=SCRIPTS_DIR)
        prompts = sample_prompts(num_prompts=NUM_PROMPTS, call_responses=call_responses, response_emotion='neutral')

    if SAVE_PROMPTS:
        print("Saving prompts...")
        if not os.path.exists(TEST_SET_DIR):
            os.makedirs(TEST_SET_DIR)
        test_set = {
            'NUM_PROMPTS': NUM_PROMPTS,
            'PROMPTS': prompts,
            'EMOTION': 'neutral'
        }
        with open(TEST_SET_DIR + f"/test_set__{NUM_PROMPTS}_prompts.json", "w") as f:
            json.dump(test_set, f)

    # Get list of generated call responses
    generated = []
    for i, p in enumerate(prompts):
        generated += generate_samples(dialogue_model, dialogue_tokenizer, p, num_samples=NUM_SAMPLES)
        print(f"Generated {i + 1}/{len(prompts)}")

    # Organize call responses in dictionary
    generated_call_responses = []
    generated_original = []
    for g in generated:
        generated_call = g.split("\n")[0]
        generated_response = g.split("\n")[1]

        if TARGET_EMOTION == "":
            generated_call_response = {
                "call": {"emotion": generated_call.split(":")[1].strip(" ()"),
                         "text": generated_call.split(":")[-1].strip()},
                "response": {"emotion": generated_response.split(":")[1].strip(" ()"),
                             "text": generated_response.split(":")[-1].strip()}
            }
        else:
            generated_call_response = {
                "call": {"emotion": TARGET_EMOTION, "text": generated_call.split(":")[-1].strip()},
                "response": {"emotion": TARGET_EMOTION, "text": generated_response.split(":")[-1].strip()}
            }

        # Add if not empty
        if generated_call_response["response"]["text"] != "":
            generated_call_responses.append(generated_call_response)
            generated_original.append(g)

    print("Evaluating...")
    report, mean_averages, labels = evaluate(emotion_model, emotion_tokenizer, generated_call_responses)

    print("Saving results...")
    labeled_generations = ""
    for i, g in enumerate(generated_original):
        labeled_generations += g + "\n"
        labeled_generations += "Labeled: " + labels[i] + "\n"
        labeled_generations += "---\n"

    formatted_results = f"RESULTS\n{report}\n\nGENERATIONS\n{labeled_generations}"

    if not os.path.exists(SAVE_RESULTS_DIR):
        os.makedirs(SAVE_RESULTS_DIR)

    with open(SAVE_RESULTS_DIR + "/results.txt", "w") as f:
        f.write(formatted_results)

    print("Done.")


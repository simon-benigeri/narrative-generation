import os
import re
import json
import numpy as np

with open("config.yml", "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    
# Softmax emotion dict if currently in score values
def softmax_emotions(emotion_dict):
    scores = np.array(list(emotion_dict.values()))
    probs = np.exp(scores) / np.sum(np.exp(scores))
    return dict(zip(list(emotion_dict.keys()), probs.tolist()))

# Get emotion from emotion dict
def get_emotion(emotion_dict, threshold=EMOTION_THRESHOLD):
    scores=list(softmax_emotions(emotion_dict).values())
    if max(scores)>threshold:
        return list(emotion_dict.keys())[scores.index(max(scores))]
    return "neutral"

def get_emotion_scores(text, emotions=config.EMOTIONS):
    input_ids = tokenizer.encode(text + '</s>', return_tensors='pt')
    output = model.generate(input_ids=input_ids, max_length=2, return_dict_in_generate=True,  output_scores=True)

    # Get emotion label scores
    emotion_scores = [
        output.scores[0][0][tokenizer.encode(e)[0]].item() for e in emotions
    ]

    dec = [tokenizer.decode(ids) for ids in output.sequences]

    return dict(zip(emotions, list(map(float, emotion_scores))))
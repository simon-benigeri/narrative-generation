import os
import re
import json
import numpy as np
import yaml

with open("config.yml", "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    
""" Softmax emotion dict if currently in score values """
def softmax_emotions(emotion_dict):
    scores = np.array(list(emotion_dict.values()))
    probs = np.exp(scores) / np.sum(np.exp(scores))
    return dict(zip(list(emotion_dict.keys()), probs.tolist()))

""" Get emotion from emotion dict """
def get_emotion(emotion_dict, threshold):
    scores=list(softmax_emotions(emotion_dict).values())
    if max(scores)>threshold:
        return list(emotion_dict.keys())[scores.index(max(scores))]
    return "neutral"

""" Get emotion scores for text with specified emotions from T5 model """
def get_emotion_scores(text, model, tokenizer, emotions=config["EMOTIONS"]):
    input_ids = tokenizer.encode(text + '</s>', return_tensors='pt')
    output = model.generate(input_ids=input_ids, max_length=30, return_dict_in_generate=True,  output_scores=True)

    # Trim to max length
    if len(input_ids[0]) > 512:
        input_ids = input_ids[:,:512]
        
    # Get emotion label scores
    emotion_scores = [
        output.scores[0][0][tokenizer.encode(e)[0]].item() for e in emotions
    ]

    dec = [tokenizer.decode(ids) for ids in output.sequences]

    return dict(zip(emotions, list(map(float, emotion_scores))))

""" Generated text from model given prompt and number of desired samples """
def generate_samples(model, tokenizer, prompt, num_samples=1):
    model.eval()
    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(model.device)

    sample_outputs = model.generate(
        generated,
        # bos_token_id=random.randint(1,30000),
        do_sample=True,
        top_k=50,
        max_length=300,
        top_p=0.90,
        num_return_sequences=num_samples
    )

    return [tokenizer.decode(sample_output, skip_special_tokens=True) for sample_output in sample_outputs]

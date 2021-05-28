import os
import numpy as np
import pandas as pd
import re
import os
import torch
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForSeq2SeqLM, GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

EMOTIONS = ['joy', 'love', 'fear', 'sadness', 'anger', 'surprise']
LOAD_DIALOGUE_MODEL_DIR = os.environ.get('LOAD_DIALOGUE_MODEL_DIR', 'models/temp/model_save')
SAVE_RESULTS_DIR = os.environ.get('SAVE_RESULTS_DIR', 'models/temp/results')
THRESHOLD = float(os.environ.get('THRESHOLD', 0.7))
TARGET_EMOTION = os.environ.get('TARGET_EMOTION', "")
NUM_SAMPLES = int(os.environ.get('NUM_SAMPLES', 1))

def get_emotion_scores(model, tokenizer, text):
    input_ids = tokenizer.encode(text + '</s>', return_tensors='pt')
    output = model.generate(input_ids=input_ids, max_length=2, return_dict_in_generate=True,  output_scores=True)

    # Get emotion label scores
    emotions = ['joy', 'love', 'fear', 'sadness', 'anger', 'surprise']
    emotion_scores = [
        output.scores[0][0][tokenizer.encode('joy')[0]].item(),
        output.scores[0][0][tokenizer.encode('love')[0]].item(),
        output.scores[0][0][tokenizer.encode('fear')[0]].item(),
        output.scores[0][0][tokenizer.encode('sadness')[0]].item(),
        output.scores[0][0][tokenizer.encode('anger')[0]].item(),
        output.scores[0][0][tokenizer.encode('surprise')[0]].item()
    ]

    dec = [tokenizer.decode(ids) for ids in output.sequences]
    label = re.sub(r'\<[^)]*\>', '', dec[0]).strip()
    scores = list(map(float,list(torch.nn.functional.softmax(torch.tensor(emotion_scores), dim=0).detach().numpy())))

    return scores


def evaluate(model, tokenizer, samples, emotion_tags_included=True, target_response_emotion=""):
    outputs = np.array([list(_get_emotion(model, tokenizer, sample)) for sample in samples])
    df = pd.DataFrame(data=outputs,
                      columns = ['target_labels', 'target_confidence', 'predicted_labels', 'predicted_confidence'])
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

    report = classification_report(y_true, y_pred, labels=EMOTIONS, target_names=EMOTIONS)
    mean_averages = df.groupby('target_labels')['target_confidence'].mean()

    return report, mean_averages, y_pred

def _get_emotion(model, tokenizer, call_response):
    target = call_response["response"]["emotion"]
    response = call_response["response"]["text"]
    
    try:
      scores = get_emotion_scores(model, tokenizer, response)

      predicted = EMOTIONS[scores.index(max(scores))] if max(scores) > THRESHOLD else "neutral"
      target_confidence = scores[EMOTIONS.index(target)]
      predicted_confidence = max(scores) if max(scores) > THRESHOLD else 1 - max(scores)

    except Exception as e:
      print('Error with : ', response, " : ", e)
      predicted = None
      target_confidence = None
      predicted_confidence = None

    return (target, target_confidence, predicted, predicted_confidence)

def generate_samples(model, tokenizer, prompt, num_samples=1):
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

def get_call_response_dict(generated, target_emotion="", allow_empty=False):
    generated_call_responses = []
    generated_original = []
    for g in generated:
        generated_call = g.split("\n")[0]
        generated_response = g.split("\n")[1]

        if target_emotion == "":
            generated_call_response = {
              "call":{"emotion": generated_call.split(":")[1].strip(" ()"), "text": generated_call.split(":")[-1].strip()},
              "response":{"emotion": generated_response.split(":")[1].strip(" ()"), "text": generated_response.split(":")[-1].strip()}
            }
        else:
            generated_call_response = {
              "call":{"emotion": "neutral", "text": generated_call.split(":")[-1].strip()},
              "response":{"emotion": target_emotion, "text": generated_response.split(":")[-1].strip()}
            }
        
        # Add if not empty
        if generated_call_response["response"]["text"] != "" or allow_empty:
          generated_call_responses.append(generated_call_response)
          generated_original.append(g)
        
    return generated_call_responses, generated_original

def emotion_tag_prompts(prompts, call_response_prompt_dict):
  tagged_prompts = []
  for prompt in call_response_prompt_dict:
      for e in EMOTIONS:
          tagged_prompts.append(f"C: (neutral): {prompt['call']['text']}\nR: ({e}): {prompt['response']['text']}")
  return tagged_prompts

if __name__=='__main__':
    print("Loading models...")
    print("Loading T5 emotion tagger...")
    # Load emotion model
    emotion_tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")
    emotion_model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-emotion")
    emotion_model.resize_token_embeddings(len(emotion_tokenizer))
    emotion_model.eval()
    
    # Load dialogue model
    print("Loading GPT2 dialogue...")
    dialogue_tokenizer = GPT2Tokenizer.from_pretrained(LOAD_DIALOGUE_MODEL_DIR, bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
    configuration = GPT2Config.from_pretrained(LOAD_DIALOGUE_MODEL_DIR, output_hidden_states=False)
    dialogue_model = GPT2LMHeadModel.from_pretrained(LOAD_DIALOGUE_MODEL_DIR, config=configuration)
    dialogue_model.resize_token_embeddings(len(dialogue_tokenizer))
    dialogue_model.eval()

    print("Generating...")

    prompts = [
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
    
    prompts = emotion_tag_prompts(prompts, get_call_response_dict(prompts, target_emotion="", allow_empty=True)[0])

    # Get list of generated call responses
    generated = []
    for i, p in enumerate(prompts):
        generated += generate_samples(dialogue_model, dialogue_tokenizer, p, num_samples=NUM_SAMPLES)
        print(f"Generated {i+1}/{len(prompts)}")

    # Organize call responses in dictionary
    generated_call_responses = []
    generated_original = []
    for g in generated:
        generated_call = g.split("\n")[0]
        generated_response = g.split("\n")[1]

        if TARGET_EMOTION == "":
            generated_call_response = {
              "call":{"emotion": generated_call.split(":")[1].strip(" ()"), "text": generated_call.split(":")[-1].strip()},
              "response":{"emotion": generated_response.split(":")[1].strip(" ()"), "text": generated_response.split(":")[-1].strip()}
            }
        else:
            generated_call_response = {
              "call":{"emotion": TARGET_EMOTION, "text": generated_call.split(":")[-1].strip()},
              "response":{"emotion": TARGET_EMOTION, "text": generated_response.split(":")[-1].strip()}
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






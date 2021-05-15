"""
Process raw script by annotating script lines (emotions, intentions, etc.) and save
"""

import os
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

RAW_SCRIPTS_DIR = "data/raw/ScreenPyOutput"
PROCESSED_SCRIPTS_DIR = "data/processed/json"
EMOTION_SCORE_THRESHOLD = 4

def get_emotion(text):
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
    return dict(zip(emotions, list(map(float,list(torch.nn.functional.softmax(torch.tensor(emotion_scores), dim=0).detach().numpy())))))

def process_screenpy_script(script_dict, title, genre):
    # Process and append each stage direction and utterance in each scene into a new script
    processed_script_dict = {
        "title":title,
        "genre":genre,
        "scenes":[]
    }
    for scene in script_dict:
        processed_scene_lines = []
        skip_next = False
        for i, line in enumerate(scene):
            # Skip this line
            if skip_next:
                skip_next = False
                continue

            # Process and add stage direction if text is not empty
            if line["head_type"] == "heading" and line["text"] != "":
                processed_scene_lines.append({
                    "type":"stage_direction",
                    "text":line["text"]
                })

            # Process and add utterance if text is not empty
            if line["head_type"] == "speaker/title" and line["text"] != "":
                # Clean character_name and text (remove parenthesis and extra spaces)
                character_name = re.sub(r'\([^)]*\)', '', line["head_text"]["speaker/title"]).strip()
                text = re.sub(r'\([^)]*\)', '', line["text"]).strip()

                # Append text to previous line if character name is empty
                if character_name == "" and len(processed_scene_lines) > 1:
                    processed_scene_lines[-1]["text"] += text
                    continue 

                # Use text from the following line if this one is empty
                if text == "" and i < len(scene) - 1:
                    text = re.sub(r'\([^)]*\)', '', scene[i+1]["text"]).strip()
                    skip_next = True

                processed_scene_lines.append({
                    "type":"utterance",
                    "character":character_name,
                    "emotion": get_emotion(line["text"]),
                    "text":text
                })

        # Add scene to processed script
        processed_script_dict["scenes"].append({
            "num":len(processed_scene_lines),
            "characters":None,
            "goals":None,
            "outcomes":None,
            "lines":processed_scene_lines
        })
    
    return processed_script_dict
    

def is_screenpy_script_valid(script_dict):
    # Skip if empty script
    if len(script_dict) == 0:
        return False
    
    num_heading = 0
    num_speaker = 0
    # Loop through script for other errors
    for scene in script_dict:
        for line in scene:
            if line["head_type"] == "heading": num_heading += 1
            if line["head_type"] == "speaker/title": num_speaker += 1

    # Skip if either heading or speaker is 0
    if num_heading == 0 or num_speaker == 0:
        return False
    
    # All good if we get here
    return True
    
def save_script_dict(dir, filename, script_dict):
    if not ".json" in filename:
        filename += ".json"

    with open(os.path.join(dir, filename), 'w') as f:
        json.dump(script_dict, f)

def process_screenpy_scripts(load_dir, save_dir, process_genres=[]):
    num_processed = 0
    num_skipped = 0
    # Loop through every genre in every subdirectory
    for subdir, dirs, files in os.walk(load_dir):
        genre = subdir.split("/")[-1]

        # Skip genres not in the process_genres list (process all if empty)
        if len(process_genres) > 0 and not genre in process_genres:
            continue
        
        # Get already processed files
        already_processed = []
        for subdir_ap, dirs_ap, files_ap in os.walk(os.path.join(save_dir, genre)):
            already_processed += files_ap

        for file in files:
            # Skip if already processed
            if file in already_processed:
                print("Script has already been processed. Skipping...")
                continue

            # Stop early for testing
            if num_processed == 2:
                return

            # Read all script json
            if file.endswith('.json'):
                with open(os.path.join(subdir, file), 'r') as f:
                    script_dict = json.load(f)

                # Process script and save it if it's valid
                if is_screenpy_script_valid(script_dict):
                    print("Processing {}...".format(file))
                    num_processed += 1
                    processed_script_dict = process_screenpy_script(script_dict, file.split(".")[0], genre)

                    print("Saving...")
                    save_script_dict(os.path.join(save_dir, genre), file, processed_script_dict)
                else:
                    print("Skipping {}...".format(file))
                    num_skipped += 1

    print("Done. Processed {} and skipped {}".format(num_processed, num_skipped))

if __name__ == '__main__':
    # Load models to label emotions
    #global tokenizer
    #global model
    #tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")
    #model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-emotion")

    #process_screenpy_scripts(RAW_SCRIPTS_DIR, PROCESSED_SCRIPTS_DIR, process_genres=["Action"])
    with open(os.path.join(RAW_SCRIPTS_DIR + "/Action/avatar.json"), 'r') as f:
        script_dict = json.load(f)
    
    sd = process_screenpy_script(script_dict, "...", "Action")

    for s in sd["scenes"]:
        for l in s["lines"]:
            print(l)

    





    
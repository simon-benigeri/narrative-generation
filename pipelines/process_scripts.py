"""
Process raw script by annotating script lines (emotions, intentions, etc.) and save
"""

import os
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

RAW_SCRIPTS_DIR = os.environ.get('RAW_SCRIPTS_DIR', "data/raw/ScreenPyOutput")
PROCESSED_SCRIPTS_DIR = os.environ.get('PROCESSED_SCRIPTS_DIR', "data/processed/json")
PROCESS_GENRES = os.environ.get('PROCESS_GENRES', "Action").split(" ") # Genres to process (seperated by a space)
MAX_SCRIPTS_PROCESS = int(os.environ.get('MAX_SCRIPTS_PROCESS', "0"))
EMOTION_SCORE_THRESHOLD = 4

# https://www.calmsage.com/understanding-the-emotion-wheel/

EMOTIONS_6 = ["joy", "love", "anger", "fear", "sadness", "surprise"]

EMOTIONS_PLUTCHIK_42 = [
  # Happy
  "playful", "content", "interested", "proud", "accepted", "powerful", "peaceful", "trusting", "optimistic",
  # Sad
  "lonely", "vulnerable", "despair", "guilty", "depressed", "hurt",
  # Disgusted
  "repelled", "awful", "disappointed", "disapproving",
  # Angry
  "critical", "distant", "frustrated", "aggressive", "mad", "bitter", "humiliated", "disillusioned",
  # Fearful
  "threatened", "rejected", "weak", "insecure", "anxious", "scared",
  # Bad
  "bored", "busy", "stressed", "tired",
  # Surprised
  "startled", "confused", "amazed", "excited"
]

EMOTIONS_PLUTCHIK_84 = [
  # "betrayed", "resentful", "disrespected", "ridiculed", "indignant", "violated", "furious", "jealous", "provoked", "hostile", "infuriated", "annoyed", "withdrawn", "numb", "skeptical", "dismissive",
]

def get_emotion(text, emotions=EMOTIONS_6):
    input_ids = tokenizer.encode(text + '</s>', return_tensors='pt')
    output = model.generate(input_ids=input_ids, max_length=2, return_dict_in_generate=True,  output_scores=True)

    # Get emotion label scores
    emotion_scores = [
        output.scores[0][0][tokenizer.encode(e)[0]].item() for e in emotions
    ]

    dec = [tokenizer.decode(ids) for ids in output.sequences]

    return dict(zip(emotions, list(map(float, emotion_scores))))

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
    
    # Create output directory if needed
    if not os.path.exists(dir):
        os.makedirs(dir)

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

            # Cap number of scripts to process
            if MAX_SCRIPTS_PROCESS > 0 and num_processed >= MAX_SCRIPTS_PROCESS:
                print("Done. Stopping processing at {} scripts.".format(MAX_SCRIPTS_PROCESS))
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
    global tokenizer
    global model
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")
    model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-emotion")

    process_screenpy_scripts(RAW_SCRIPTS_DIR, PROCESSED_SCRIPTS_DIR, process_genres=PROCESS_GENRES)
    
    """
    with open(os.path.join(RAW_SCRIPTS_DIR + "/Action/avatar.json"), 'r') as f:
        script_dict = json.load(f)
    
    sd = process_screenpy_script(script_dict, "...", "Action")

    for s in sd["scenes"]:
        for l in s["lines"]:
            print(l)
    """

    





    
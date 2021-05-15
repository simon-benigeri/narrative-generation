"""
Load up processed scrips and format data for training
"""
import os
import re
import json

EMOTION_THRESHOLD = 0.7
JSON_SCRIPTS_DIR = "data/processed/json/Action"
FORMATTED_SCRIPTS_DIR = "data/processed/formatted/scripts/Action"
FORMATTED_CALL_RESPONSE_DIR = "data/processed/formatted/call_responses/Action"

START_LINE_TAG = "<start-line>"
END_LINE_TAG = "<end-line>\n"

NEW_CALL_RESPONSE_TAG = "---\n"

# Get emotion from emotion dict
def get_emotion(emotion_dict, threshold=EMOTION_THRESHOLD):
  scores=list(emotion_dict.values())
  if max(scores)>threshold:
    return list(emotion_dict.keys())[scores.index(max(scores))]
  return "neutral"

# Format call response
def format_call_response(script_dict, include_character=True, include_emotion=True):
    call_responses = []
    for scene in script_dict["scenes"]:
        for i, line in enumerate(scene["lines"]):
            formatted_call_response = ""
            if line["type"] == "utterance" and i < len(scene["lines"]) - 1 and scene["lines"][i+1]["type"] == "utterance":
                # Call
                call_character = f"{line['character']}: " if include_character else "C: "
                call_emotion = f"({get_emotion(line['emotion'], threshold=EMOTION_THRESHOLD)}): " if include_emotion else ""
                formatted_call_response += f"{call_character}{call_emotion}{line['text']}\n"

                # Response
                response_character = f"{scene['lines'][i+1]['character']}: " if include_character else "R: "
                response_emotion = f"({get_emotion(scene['lines'][i+1]['emotion'], threshold=EMOTION_THRESHOLD)}): " if include_emotion else ""
                formatted_call_response += f"{response_character}{response_emotion}{scene['lines'][i+1]['text']}\n"

                call_responses.append(formatted_call_response)
    
    return NEW_CALL_RESPONSE_TAG.join(call_responses)

# Format individual json script
def format_script(script_dict, include_emotion=True):
    formatted_script = []
    for scene in script_dict["scenes"]:
        formatted_scene = "<start-context>"
        initial_context = False
        num_utterances = 0
        for i, line in enumerate(scene["lines"]):
            # If the first line is stage direction, then treat it as context
            if line["type"] == "stage_direction":
                if i == 0:
                    initial_context = True
                    formatted_scene += line["text"]
                elif i > 0 and scene["lines"][i-1]["type"] == "stage_direction":
                    if i < len(scene["lines"]) - 1 and scene["lines"][i+1]["type"] == "stage_direction":
                        formatted_scene += " " + line["text"]
                    else:
                        formatted_scene += " " + line["text"] + END_LINE_TAG
                elif i > 0 and scene["lines"][i-1]["type"] == "utterance":
                    formatted_scene += START_LINE_TAG + "DIRECTION: " + line["text"]
                    if i < len(scene["lines"]) - 1 and scene["lines"][i+1]["type"] == "utterance":
                        formatted_scene += END_LINE_TAG

            # Format utterance
            if line["type"] == "utterance":
                if not "<end-context>" in formatted_scene:
                    formatted_scene += "<end-context>\n<start-dialogue>"
                
                # Format utterance with everything in paranthesis in utterance removed
                formatted_scene += START_LINE_TAG + line["character"] + ": "
                if include_emotion:
                    formatted_scene += "(" + get_emotion(line["emotion"]) + ") : "
                formatted_scene += re.sub(r'\([^)]*\)', '', line["text"]) + END_LINE_TAG
                num_utterances += 1
        
        # Remove last new line
        if formatted_scene[-1] == "\n":
            formatted_scene = formatted_scene[:-1]

        formatted_scene += "<end-dialogue>"
        
        # Append formatted scene to formatted script if it had context and dialogue
        if initial_context and num_utterances > 1:
            formatted_script.append(formatted_scene)
    
    return "\n<new-scene>\n".join(formatted_script)

def save_formatted_script(dir, filename, formatted_script):
    if not ".txt" in filename:
        filename += ".txt"

    with open(os.path.join(dir, filename), 'w') as f:
        f.write(formatted_script)

def format_scripts(load_dir, save_dir, format_type="whole_script", include_character=True, include_emotion=True):
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load, format, and save scripts
    for subdir, dirs, files in os.walk(load_dir):
        for file in files:
            if file == ".DS_Store": continue

            print("formatting {}".format(file))

            # Load json script
            with open(os.path.join(subdir, file), 'r') as f:
                script_dict = json.load(f)
            
            # Format script
            if format_type == "whole_script":
                formatted_script = format_script(script_dict, include_emotion=include_emotion)
            elif format_type == "call_response":
                formatted_script = format_call_response(script_dict, include_character=include_character, include_emotion=include_emotion)

            # Save script
            save_formatted_script(save_dir, file.split(".")[0], formatted_script)
    print("Done")

if __name__ == '__main__':
    format_scripts(JSON_SCRIPTS_DIR, FORMATTED_SCRIPTS_DIR, format_type="call_response", include_character=False, include_emotion=True)

    """
    with open(os.path.join(JSON_SCRIPTS_DIR, "kungfupanda.json"), 'r') as f:
        script_dict = json.load(f)

    print(format_call_response(script_dict, include_character=True, include_emotion=True))
    """
    
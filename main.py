from system.NarrativeController import NarrativeController
from system.Emotions import Emotions
from system.Script import Script
import argparse
from typing import List

parser = argparse.ArgumentParser(description="Narrative Generation")
parser.add_argument("--arcX", type=List[str], default=["anger"], help="Emotional arc for Character X.")
parser.add_argument("--arcY", type=List[str], default=["surprise, anger, fear"], help="Emotional arc for Character Y.")
parser.add_argument("--dirX", type=List[str], default=["Character_X is holding a knife", "Character_X is standing up"], help="Initial directions for Character X.")
parser.add_argument("--dirY", type=List[str], default=["Character_Y in front of Character_X"], help="Initial directions for Character Y.")
args = vars(parser.parse_args())

if __name__ =='__main__':
    print("Initializing narrative contoller...")
    narrative_controller = NarrativeController()

    short_arc = [Emotions.HAPPINESS, Emotions.FEAR]
    long_arc = [Emotions.HAPPINESS, Emotions.FEAR, Emotions.SADNESS, Emotions.ANGER, Emotions.HAPPINESS]

    # Define story starting point...
    initial_script = Script(arc=short_arc)

    initial_script.append_direction(direction="Once upon a time, there lived a man named character x")
    initial_script.append_direction(direction="He lived a happy life")
    initial_script.append_utterance(utterance="I am very happy!", character="Character x")

    initial_script.arc_step()

    # Generate the rest of the script
    print("Generating script...")
    generated_script = narrative_controller.generate_script(arc=initial_script.arc, initial_script=initial_script)
    
    print("\nGENERATED SCRIPT")
    print(generated_script)

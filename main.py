from system.NarrativeController import NarrativeController
from system.Emotions import Emotions
from system.Character import Character
from system.Script import Script
import argparse
from typing import List

"""
parser = argparse.ArgumentParser(description="Narrative Generation")
parser.add_argument("--arcX", type=List[str], default=["anger"], help="Emotional arc for Character X.")
parser.add_argument("--arcY", type=List[str], default=["surprise, anger, fear"], help="Emotional arc for Character Y.")
parser.add_argument("--dirX", type=List[str], default=["Character_X is holding a knife", "Character_X is standing up"], help="Initial directions for Character X.")
parser.add_argument("--dirY", type=List[str], default=["Character_Y in front of Character_X"], help="Initial directions for Character Y.")
args = vars(parser.parse_args())
"""

if __name__ == '__main__':
    
    # Define characters
    alice = Character(name="Alice", arc=[Emotions.ANGER, Emotions.ANGER])
    bob = Character(name="Bob", arc=[Emotions.HAPPINESS, Emotions.FEAR])
    
    # Initialize narrative controller
    print("Initializing narrative controller...")
    narrative_controller = NarrativeController(characters=[alice, bob])
    
    # Define initial script
    narrative_controller.script.append_direction(direction="Alice became very angry.")
    narrative_controller.script.append_direction(direction="Alice grabbed a knife from the kitchen and went to the park.")
    narrative_controller.script.append_direction(direction="That's where Bob was eating lunch.")
    narrative_controller.script.append_utterance(utterance="I will find Bob!", character=alice)
    narrative_controller.arc_step()

    # Generate the rest of the script
    print("Generating script...")
    generated_script = narrative_controller.generate_script()
    
    print("\n ============ GENERATED SCRIPT ============ \n")
    print(str(generated_script))

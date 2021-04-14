from system.NarrativeController import NarrativeController
from system.Emotions import Emotions
import argparse
from typing import List

parser = argparse.ArgumentParser(description="Narrative Generation")
parser.add_argument("--arcX", type=List[str], default=["anger"], help="Emotional arc for Character X.")
parser.add_argument("--arcY", type=List[str], default=["surprise, anger, fear"], help="Emotional arc for Character Y.")
parser.add_argument("--dirX", type=List[str], default=["Character_X is holding a knife", "Character_X is standing up"], help="Initial directions for Character X.")
parser.add_argument("--dirY", type=List[str], default=["Character_Y in front of Character_X"], help="Initial directions for Character Y.")
args = vars(parser.parse_args())

if __name__ =='__main__':
    narrative_controller = NarrativeController()

    
    print(narrative_controller.generate_script(arc))

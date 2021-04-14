from .Script import Script
from .Emotions import Emotions

"""
Component to generate dialogue between characters
"""
class DialogueGenerator:
    def __init__(self):
        # Load all transformers
        # self.happy_transformer = ...
        # ...

        # Dictionary for emotion and corresponding transformer
        self.transformers = {
            # Emotions.HAPPINESS: self.happy_transformer,
            # ...
        }
    
    # Generate next character utterances given direction and past utterences
    def generate_dialogue(self, script:Script):
        # Generate dialogue
        # ...

        return ["X:what emotion are you?", "Y:im happy"]

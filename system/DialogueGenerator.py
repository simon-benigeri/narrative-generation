

"""
Component to generate dialogue between characters
"""
class DialogueGenerator:
    def __init__(self):
        # Load all transformers
        self.happy_transformer = ...
        ...

        # Dictionary for emotion and corresponding transformer
        self.transformers = {
            "HAPPY": self.happy_transformer,
            ...
        }
    
    # Generate next character utterances given direction and past utterences
    def generate_utterances(self, emotion:str, prev_direction:str, prev_utterance:str):
        # Generate utterences
        # ...

        return utterences
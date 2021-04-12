

class DialogueGenerator:
    def __init__(self):
        self.transformers = {
            "HAPPY": self.happy_transformer,
            ...
        }
    
    def generate(self, emotion):
        # generate based on emotion
        return
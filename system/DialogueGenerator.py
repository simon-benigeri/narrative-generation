from .Script import Script
from .Emotions import Emotions

import torch
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

"""
Component to generate dialogue between characters
"""
class DialogueGenerator:
    def __init__(self):
        dialogue_models_dir = "saved_models/dialogue_models/"

        # Load tokenizer and set configuration for transformers
        self.tokenizer = GPT2Tokenizer.from_pretrained(dialogue_models_dir + "fear_transformer", bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
        configuration = GPT2Config.from_pretrained(dialogue_models_dir + "fear_transformer", output_hidden_states=False)

        # Initialize all transformers...

        # TEMP
        self.fear_transformer = GPT2LMHeadModel.from_pretrained(dialogue_models_dir + "fear_transformer", config=configuration)
        self.fear_transformer.resize_token_embeddings(len(self.tokenizer))
        self.fear_transformer.eval()

        self.transformers = {
            Emotions.FEAR: self.fear_transformer,
        }

        # Ignore them for now
        return

        # Happiness transformer
        self.happiness_transformer = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
        self.happiness_transformer.resize_token_embeddings(len(self.tokenizer))
        self.happiness_transformer.eval()

        # Sadness transformer
        self.sadness_transformer = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
        self.sadness_transformer.resize_token_embeddings(len(self.tokenizer))
        self.sadness_transformer.eval()

        # Fear transformer
        self.fear_transformer = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
        self.fear_transformer.resize_token_embeddings(len(self.tokenizer))
        self.fear_transformer.eval()

        # Disgust transformer
        self.disgust_transformer = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
        self.disgust_transformer.resize_token_embeddings(len(self.tokenizer))
        self.disgust_transformer.eval()

        # Anger transformer
        self.anger_transformer = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
        self.anger_transformer.resize_token_embeddings(len(self.tokenizer))
        self.anger_transformer.eval()

        # Surprise transformer
        self.surprise_transformer = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
        self.surprise_transformer.resize_token_embeddings(len(self.tokenizer))
        self.surprise_transformer.eval()

        # Dictionary for emotion and corresponding transformer
        self.transformers = {
            Emotions.HAPPINESS: self.happiness_transformer,
            Emotions.SADNESS: self.sadness_transformer,
            Emotions.FEAR: self.fear_transformer,
            Emotions.DISGUST: self.disgust_transformer,
            Emotions.ANGER: self.anger_transformer,
            Emotions.SURPRISE: self.surprise_transformer,
        }
    
    # Generate next character utterances given direction and past utterences
    def generate_dialogue(self, script:Script):
        # TODO: Need to associate characters with text

        # Hard code emotion and input lines for now
        emotion = Emotions.FEAR
        prompt_lines = "<|startoftext|> C: "

        generated = torch.tensor(self.tokenizer.encode(prompt_lines)).unsqueeze(0)
        #generated = generated.to(device)
        generated = self.transformers[emotion].generate(
                                        generated, 
                                        #bos_token_id=random.randint(1,30000),
                                        do_sample=True,
                                        top_k=50,
                                        max_length = 300,
                                        top_p=0.90,
                                        num_return_sequences=2
                                    )

        # Not currently a list of sequential utterances
        # Decode into list of strings
        generated = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated]

        # Do something to associate characters...

        # Return list of character, utteance pairs
        character_utterances = []
        for g in generated:
            character_utterances.append(("Character ?", g))
        
        return character_utterances

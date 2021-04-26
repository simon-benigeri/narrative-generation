from .Script import Script
from .Emotions import Emotions
from .Character import Character

import torch
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

"""
Component to generate dialogue between characters
"""
class DialogueGenerator:
    def __init__(self):
        dialogue_models_dir = "saved_models/dialogue_models/"

        # Load tokenizer and set configuration for transformers
        self.tokenizer = GPT2Tokenizer.from_pretrained(dialogue_models_dir + "fear_transformer", bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
        configuration = GPT2Config.from_pretrained(dialogue_models_dir + "fear_transformer", output_hidden_states=False)

        # Initialize all transformers
        self.happiness_transformer = GPT2LMHeadModel.from_pretrained(dialogue_models_dir + "fear_transformer", config=configuration)
        self.happiness_transformer.resize_token_embeddings(len(self.tokenizer))
        self.happiness_transformer.eval()

        self.sadness_transformer = GPT2LMHeadModel.from_pretrained(dialogue_models_dir + "fear_transformer", config=configuration)
        self.sadness_transformer.resize_token_embeddings(len(self.tokenizer))
        self.sadness_transformer.eval()

        self.fear_transformer = GPT2LMHeadModel.from_pretrained(dialogue_models_dir + "fear_transformer", config=configuration)
        self.fear_transformer.resize_token_embeddings(len(self.tokenizer))
        self.fear_transformer.eval()

        self.disgust_transformer = GPT2LMHeadModel.from_pretrained(dialogue_models_dir + "fear_transformer", config=configuration)
        self.disgust_transformer.resize_token_embeddings(len(self.tokenizer))
        self.disgust_transformer.eval()

        self.anger_transformer = GPT2LMHeadModel.from_pretrained(dialogue_models_dir + "fear_transformer", config=configuration)
        self.anger_transformer.resize_token_embeddings(len(self.tokenizer))
        self.anger_transformer.eval()

        self.surprise_transformer = GPT2LMHeadModel.from_pretrained(dialogue_models_dir + "fear_transformer", config=configuration)
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
    
    """ Generate next character utterances given direction and past utterences """
    def generate_dialogue(self, script:Script, context_text:str, leading_character:Character, response_character:Character) -> str:

        # TEMP: Hard code emotion and input lines for now
        emotion = leading_character.current_emotion()

        # Prompt generator with context text
        prompt = "<|startoftext|>" + context_text + " C: "

        generated = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0)
        #generated = generated.to(device)
        generated = self.transformers[emotion].generate(
                                        generated,
                                        do_sample=True,
                                        top_k=50,
                                        max_length = 300,
                                        top_p=0.90,
                                        num_return_sequences=1
                                    )

        # Not currently a list of sequential utterances
        # Decode into list of strings
        generated = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated]

        # TODO: Do something to associate characters...

        # Return list of character, utteance pairs
        character_utterances = []
        for g in generated:
            # Remove promt string from generated
            g = g.replace(prompt, '')
            # TODO: hard coded to character X
            character_utterances.append((character, g))
        
        return character_utterances

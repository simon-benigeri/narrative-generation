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

    """ Clean utterance text """
    def clean_utterance(self, utterance_text):
        return utterance_text.replace('\n', '')
    
    """ Generate next character utterances given direction and past utterences """
    def generate_dialogue(self, script:Script, context_text:str, leading_character:Character, response_character:Character) -> str:
        # Get last N lines of sctipt as context
        script_text = str(script)

        # List of utterances of each character in the dialogue
        character_utterances = []

        # Generate call from leading character
        call_prompt = "<|startoftext|>" + script_text + ". " + context_text + " C: "
        generated_call = torch.tensor(self.tokenizer.encode(call_prompt)).unsqueeze(0)
        generated_call = self.transformers[leading_character.current_emotion()].generate(
                                        generated_call,
                                        do_sample=True,
                                        top_k=50,
                                        max_length = 300,
                                        top_p=0.90,
                                        num_return_sequences=1
                                    )
        generated_call = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_call][0]
        generated_call = generated_call.split("R:")[0]

        # Generate response from response character
        response_prompt = generated_call + "\nR: "
        generated_response = torch.tensor(self.tokenizer.encode(response_prompt)).unsqueeze(0)
        generated_response = self.transformers[response_character.current_emotion()].generate(
                                        generated_response,
                                        do_sample=True,
                                        top_k=50,
                                        max_length = 300,
                                        top_p=0.90,
                                        num_return_sequences=1
                                    )
        generated_response = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_response][0]

        # Append generated to character utterances
        character_utterances.append(
                (leading_character, self.clean_utterance(generated_call.replace(call_prompt, '')))
            )

        # Append generated to character utterances
        character_utterances.append(
                (response_character, self.clean_utterance(generated_response.replace(response_prompt, '')))
            )
        
        # List of character utterances
        return character_utterances

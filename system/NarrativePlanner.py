from operator import attrgetter
import torch
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
import os

from system.Script import Script
from system.Character import Character
from system.Planners import Planners
from system.Emotions import Emotions

# NOTE: How this class should work, its purpose, and existence is undecided

""" Component to generate script direction and guide story """
class NarrativePlanner:
    def __init__(self, planner: Planners=Planners.DEFAULT):
        # Load transformer
        self.planners = {
            Planners.DEFAULT: 'gpt2-large',
            Planners.ROCStories: '../models/ROCStories_full/model_save'
        }

        # Load tokenizer and set configuration for transformers
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.planners[planner],
                                                       bos_token='<|startoftext|>',
                                                       eos_token='<|endoftext|>',
                                                       pad_token='<|pad|>'
                                                       )
        configuration = GPT2Config.from_pretrained(self.planners[planner], output_hidden_states=False)

        # Initialize transformer...
        self.narrative_transformer = GPT2LMHeadModel.from_pretrained(self.planners[planner], config=configuration)
        self.narrative_transformer.resize_token_embeddings(len(self.tokenizer))
        self.narrative_transformer.eval()

    """ create prompt from prev n directions and k utterances in script """
    def retrieve_prompt(self, script:Script,):
        lines = script.get_prev_lines(n=1, type=script.DIRECTION) + script.get_prev_lines(n=2, type=script.UTTERANCE)
        lines.sort(key=attrgetter('line_num'))
        prompt = script.lines_to_str(lines)
        test_prompt = """
        Alice is eating the chocolate. Bob watches her with envy. 
        Bob : "Why are you eating chocolate?" 
        Alice : "I like chocolate." 
        """
        # return "<|startoftext|> " + test_prompt
        return '<|startoftext|> ' + prompt


    # Generate screen direction given past direction and utterences
    def generate_direction(self, script:Script, do_sample: bool=True, top_k: int=50, max_length: int=100,
                           top_p: float=0.90, num_return_sequences: int=3):
        # Generate direction
        prompt_lines = self.retrieve_prompt(script=script)
        generated = torch.tensor(self.tokenizer.encode(prompt_lines)).unsqueeze(0)
        generated = self.narrative_transformer.generate(
            generated,
            do_sample=do_sample,
            top_k=top_k,
            max_length=max_length,
            top_p=top_p,
            num_return_sequences=num_return_sequences
        )
        # Decode into list of strings
        generated = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated]
        # dummy output
        # return "(screen direction)"
        return generated

# Test
if __name__=='__main__':
    # Define characters
    alice = Character(name="Alice", arc=[Emotions.ANGER, Emotions.ANGER, Emotions.HAPPINESS])
    bob = Character(name="Bob", arc=[Emotions.HAPPINESS, Emotions.FEAR, Emotions.SADNESS])
    s = Script()
    N = 3
    utterances = ["Why are you eating chocolate?", "I like chocolate."]
    directions = ["Alice is eating the chocolate. Bob watches her with envy."]
    s.append_direction(directions[0])
    s.append_utterance(utterances[0], bob)
    s.append_utterance(utterances[1], alice)
    NP = NarrativePlanner(planner=Planners.DEFAULT)
    # NP = NarrativePlanner(planner=Planners.ROCStories)
    prompt = NP.retrieve_prompt(s)
    print(f"Prompt : \n{prompt}")
    generated = NP.generate_direction(s)
    output = []
    print("Output:\n" + 100 * '-')
    for i, g in enumerate(generated):
        print("{}: {}".format(i, g))
        print()
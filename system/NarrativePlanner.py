from operator import attrgetter
import torch
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

from system.Script import Script
from system.Planners import Planners
from system.Emotions import Emotions


"""
Component to generate script direction and guide story
"""
class NarrativePlanner:
    def __init__(self, planner: Planners=Planners.DEFAULT):
        # Load transformer
        self.planners = {
            Planners.DEFAULT: 'gpt2-large',
            Planners.ROCStories__spring2016: 'gpt2-large__ROCStories__spring2016',
            Planners.ROCStories__winter2017: 'gpt2-large__ROCStories__winter2017',
            Planners.ROCStories__full: 'gpt2-large__ROCStories__full',
            Planners.GLUCOSE: 'gpt2-large__GLUCOSE'
        }
        # narrative_models_dir = "saved_models/narrative_planner_models/"
        # path = narrative_models_dir + self.planner[planner]

        # Load tokenizer and set configuration for transformers
        # TODO: USE paths when we have saved models
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.planners[planner], bos_token='<|startoftext|>',
                                                       eos_token='<|endoftext|>', pad_token='<|pad|>')
        configuration = GPT2Config.from_pretrained(self.planners[planner], output_hidden_states=False)

        # Initialize transformer...
        self.narrative_transformer = GPT2LMHeadModel.from_pretrained(self.planners[planner], config=configuration)
        self.narrative_transformer.resize_token_embeddings(len(self.tokenizer))
        self.narrative_transformer.eval()

    def retrieve_prompt(self, script:Script):
        #TODO: Review get_prev_lines because theres a smarter way to so this
        lines = script.get_prev_lines(n=1, type=script.DIRECTION) \
                + script.get_prev_lines(n=2, type=script.UTTERANCE)
        lines.sort(key=attrgetter('line_num'))
        prompt = '\n'.join([f'{l.character} : "{l.text}"' if l.type == script.UTTERANCE else l.text for l in lines])
        # return "<|startoftext|> " + prompt
        return "<|startoftext|>" + prompt


    # Generate screen direction given past direction and utterences
    def generate_direction(self, script:Script):
        # Generate direction
        prompt_lines = self.retrieve_prompt(script=script)
        generated = torch.tensor(self.tokenizer.encode(prompt_lines)).unsqueeze(0)
        generated = self.narrative_transformer.generate(
            generated,
            do_sample=True,
            top_k=50,
            max_length=300,
            top_p=0.90,
            num_return_sequences=1
        )
        # Decode into list of strings
        generated = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated]

        # dummy output
        # return "(screen direction)"
        return generated


if __name__=='__main__':
    arc = [Emotions.SADNESS, Emotions.HAPPINESS]
    s = Script(arc)
    N = 3
    utterances = ["Why are you eating chocolate?.", "I like chocolate."]
    directions = ["Character_X is eating the chocolate. Character_Y watches her with envy."]
    print(str(s))
    s.append_direction(directions[0])
    s.append_utterance(utterances[0], s.CHARACTER_Y)
    s.append_utterance(utterances[1], s.CHARACTER_X)
    NP = NarrativePlanner()
    prompt = NP.retrieve_prompt(s)
    print(f"prompt : \n{prompt}")
    generated = NP.generate_direction(s)
    print(f"generated : \n{generated}")
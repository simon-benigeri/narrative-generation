from operator import itemgetter, attrgetter
from system.Script import Script
from system.Emotions import Emotions
from transformers import pipeline
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel


"""
Component to generate script direction and guide story
"""
class NarrativePlanner:
    def __init__(self):
        # Load transformer
        self.U = 2
        self.D = 1
        # configuration = GPT2Config.from_pretrained('gpt2-large', output_hidden_states=False)
        # self.generator = GPT2LMHeadModel.from_pretrained('gpt2-large', )
        self.generator = pipeline(task='text-generation', model='gpt2-large', framework='pt')
        pass

    def retrieve_prompt(self, script:Script):
        lines = script.get_prev_lines(n=self.D, type=script.DIRECTION)
        lines += script.get_prev_lines(n=self.U, type=script.UTTERANCE)
        lines.sort(key=attrgetter('line_num'))
        texts = [f'{line.character}  said, "{line.text}"' if line.type == script.UTTERANCE else line.text for line in lines]
        prompt = ' '.join(texts)
        return prompt

    # Generate screen direction given past direction and utterences
    def generate_direction(self, script:Script):
        # Generate direction
        # ...
        prompt = self.retrieve_prompt(script=script)
        direction = self.generator(text_inputs=prompt, return_full_text=True, max_len=50, early_stopping=True)

        return direction




if __name__=='__main__':
    arc = [Emotions.SADNESS, Emotions.HAPPINESS]
    s = Script(arc)
    N = 3
    utterances = ["Why are you eating chocolate?.", "I like chocolate."]
    directions = ["Bob is eating the chocolate."]
    print(str(s))
    s.append_direction(s.CHARACTER_X, directions[0])
    s.append_utterance(s.CHARACTER_Y, utterances[0])
    s.append_utterance(s.CHARACTER_X, utterances[1])
    NP = NarrativePlanner()
    # p = NP.retrieve_prompt(s)
    p = NP.generate_direction(s)
    print(p)
from operator import itemgetter, attrgetter
from system.Script import Script
from system.Emotions import Emotions
from transformers import pipeline

"""
Component to generate script direction and guide story
datasets: GLUCOSE from datasets import load_dataset

dataset = load_dataset("glucose")
"""
class NarrativePlanner:
    def __init__(self):
        # Load transformer
        self.U = 2
        self.D = 1
        self.generator = pipeline(task='text-generation', model='gpt2-large', framework='pt')
        pass

    def retrieve_prompt(self, script:Script):
        lines = script.get_prev_lines(n=self.D, type=script.DIRECTION)
        lines += script.get_prev_lines(n=self.U, type=script.UTTERANCE)
        lines.sort(key=attrgetter('line_num'))
        texts = [f"{line.character} : {line.text}" if line.type == script.UTTERANCE else line.text for line in lines]
        prompt = '\n'.join(texts)
        return prompt

    # Generate screen direction given past direction and utterences
    def generate_direction(self, script:Script):
        # Generate direction
        # ...
        prompt = self.retrieve_prompt(script=script)
        direction = self.generator(text_inputs=prompt, return_full_text=True, max_len=30)

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
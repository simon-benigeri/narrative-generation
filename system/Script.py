from collections import namedtuple
from typing import List
from .Emotions import Emotions

"""
Component to hold the script
"""
class Script:
    # Named tuple for a single line in the script
    ScriptLine = namedtuple('ScriptLine', ['line_num', 'arc_stage', 'type', 'character','text'])

    def __init__(self, arc: List[Emotions]):
        self.DIRECTION = 0
        self.UTTERANCE = 1
        self.CHARACTER_X = 'Character_X'
        self.CHARACTER_Y = 'Character_Y'

        # Set story arc and current arc stage
        self.arc = arc
        self.current_arc_stage = 0

        # Set story complete to false unless arc list is empty
        self.complete = len(self.arc) == 0

        # List of lines in the script
        self.script_lines = []

        # TODO: initial directions?
    
    # Increment arc stage
    def arc_step(self):
        # Only step if within arc length
        if self.current_arc_stage < len(self.arc) - 1:
            self.current_arc_stage += 1
        
        # Otherwise, set script to complete
        else:
            self.complete = True

    # Append utterance to script
    def append_utterance(self, character: str, utterance: str):
        self.script_lines.append(
            self.ScriptLine(len(self.script_lines), self.current_arc_stage, self.UTTERANCE, character, utterance))
    
    # Append direction to script
    def append_direction(self, direction: str, character: str=None):
        self.script_lines.append(
            self.ScriptLine(len(self.script_lines), self.current_arc_stage, self.DIRECTION, character, direction))

    # Get the last n utterances from the script
    def get_prev_utterances(self, n: int, character: str):
        # ...
        count, index = 0, 0
        prev_utterances = []
        for _ in range(len(self.script_lines)):
            if count == n:
                break
            index -= 1
            line = self.script_lines[index]
            if line.type == self.UTTERANCE and line.character == character:
                prev_utterances.insert(0, line)
                count += 1
        return prev_utterances
    
    # Get the last n directions from the script
    def get_prev_directions(self, n: int, character: str):
        # ...
        count, index = 0, 0
        prev_directions = []
        for _ in range(len(self.script_lines)):
            if count == n:
                break
            index -= 1
            line = self.script_lines[index]
            if line.type == self.DIRECTION and line.character == character:
                prev_directions.insert(0, line)
                count += 1
        return prev_directions

        # Get the last n directions from the script
    def get_prev_lines(self, n: int, type: int, character: str=None):
        """
        returns previous utterances or screen directions for a character
        Args:
            n: number of utterances
            type: utterance or direction
            character: X or Y

        Returns: prev n directions or utterances for a character

        """
        # ...
        count, index = 0, 0
        prev = []
        for _ in range(len(self.script_lines)):
            if count == n:
                break
            index -= 1
            line = self.script_lines[index]
            if line.type == type and (not character or line.character == character):
                prev.insert(0, line)
                count += 1
        return prev
    
    # Save script to file
    def save(self, file_path):
        pass
    
    # Convert script to string
    def __str__(self):
        script_str = ""
        for l in self.script_lines:
            if l.type == self.UTTERANCE:
                script_str += f"{l.character} : {l.text}" + "\n"
            elif l.type == self.DIRECTION:
                script_str += f"{l.text}" + "\n"
        return script_str

if __name__ =='__main__':
    arc = [Emotions.SADNESS, Emotions.HAPPINESS]
    s = Script(arc)
    N = 3
    utterances = [f"utterance {i}" for i in range(N)]
    directions = [f"direction {i}" for i in range(N)]
    for u, d in zip(utterances, directions):
        s.append_utterance(s.CHARACTER_X, u)
        s.append_utterance(s.CHARACTER_Y, u)
        s.append_direction(s.CHARACTER_X, d)
        s.append_direction(s.CHARACTER_Y, d)

    print(str(s))

    assert len(s.get_prev_utterances(n=0, character=s.CHARACTER_X)) == 0
    assert len(s.get_prev_utterances(n=0, character=s.CHARACTER_Y)) == 0
    assert len(s.get_prev_utterances(n=2, character=s.CHARACTER_X)) == 2
    assert len(s.get_prev_utterances(n=2, character=s.CHARACTER_Y)) == 2
    assert len(s.get_prev_utterances(n=N + 2, character=s.CHARACTER_X)) == N
    assert len(s.get_prev_utterances(n=N + 2, character=s.CHARACTER_Y)) == N

    assert len(s.get_prev_directions(n=0, character=s.CHARACTER_X)) == 0
    assert len(s.get_prev_directions(n=0, character=s.CHARACTER_Y)) == 0
    assert len(s.get_prev_directions(n=2, character=s.CHARACTER_X)) == 2
    assert len(s.get_prev_directions(n=2, character=s.CHARACTER_Y)) == 2
    assert len(s.get_prev_directions(n=N + 2, character=s.CHARACTER_X)) == N
    assert len(s.get_prev_directions(n=N + 2, character=s.CHARACTER_Y)) == N

    assert len(s.get_prev_lines(n=0, type=s.UTTERANCE, character=s.CHARACTER_X)) == 0
    assert len(s.get_prev_lines(n=0, type=s.UTTERANCE, character=s.CHARACTER_Y)) == 0
    assert len(s.get_prev_lines(n=2, type=s.UTTERANCE, character=s.CHARACTER_X)) == 2
    assert len(s.get_prev_lines(n=2, type=s.UTTERANCE, character=s.CHARACTER_Y)) == 2
    assert len(s.get_prev_lines(n=2, type=s.DIRECTION, character=s.CHARACTER_X)) == 2
    assert len(s.get_prev_lines(n=2, type=s.DIRECTION, character=s.CHARACTER_Y)) == 2
    assert len(s.get_prev_lines(n=N + 2, type=s.UTTERANCE, character=s.CHARACTER_X)) == N
    assert len(s.get_prev_lines(n=N + 2, type=s.DIRECTION, character=s.CHARACTER_Y)) == N

    # assert len(s.get_prev_lines(n=2, type=s.DIRECTION, character=s.CHARACTER_Y)) == 2

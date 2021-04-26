from collections import namedtuple
from typing import List, Tuple
from .Emotions import Emotions
from .Character import Character

"""
Component to hold the script
"""
class Script:
    # Named tuple for a single line in the script
    ScriptLine = namedtuple('ScriptLine', ['line_num', 'arc_stage', 'type', 'character', 'text'])

    def __init__(self):
        self.DIRECTION = 0
        self.UTTERANCE = 1

        # List of lines in the script
        self.script_lines = []

        # Flag if script is complete
        self.is_complete = False

    """ Append utterance to script """
    def append_utterance(self, utterance: str, character: Character):
        self.script_lines.append(
            self.ScriptLine(len(self.script_lines), character.arc_stage, self.UTTERANCE, character, utterance))
    
    """ Append direction to script """
    def append_direction(self, direction: str):
        self.script_lines.append(
            self.ScriptLine(len(self.script_lines), None, self.DIRECTION, None, direction))

    """ Get the last n utterances from the script """
    def get_prev_utterances(self, n: int, character: Character):
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
    
    """ Get the last n directions from the script """
    def get_prev_directions(self, n: int):
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

    """ Get the last n directions from the script """
    def get_prev_lines(self, n: int, type: int, character: Character=None):
        """
        returns previous utterances or screen directions for a character
        Args:
            n: number of utterances
            type: utterance or direction
            character: X or Y

        Returns: prev n directions or utterances for a character

        """
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
    
    """ Save script to file """
    def save(self, file_path):
        pass
    
    """ Convert list of lines to string """
    def lines_to_str(self, lines):
        lines_str = ""
        for l in lines:
            if l.type == self.UTTERANCE:
                lines_str += f"{l.character} : {l.text}" + "\n"
            elif l.type == self.DIRECTION:
                lines_str += f"{l.text}" + "\n"
        return lines_str

    """ Convert script to string """
    def __str__(self):
        return self.lines_to_str(self.lines)

if __name__ =='__main__':
    arc = [Emotions.SADNESS, Emotions.HAPPINESS]
    s = Script(arc)
    N = 3
    utterances = [f"utterance {i}" for i in range(N)]
    directions = [f"direction {i}" for i in range(N)]
    for u, d in zip(utterances, directions):
        s.append_utterance(u, s.CHARACTER_X)
        s.append_utterance(u, s.CHARACTER_Y)
        s.append_direction(d, s.CHARACTER_X)
        s.append_direction(d, s.CHARACTER_Y)

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

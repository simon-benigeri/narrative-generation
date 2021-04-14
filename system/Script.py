from collections import namedtuple
from typing import List

"""
Component to hold the script
"""
class Script:
    # Named tuple for a single line in the script
    ScriptLine = namedtuple('ScriptLine', ['line_num', 'arc_stage', 'type', 'text'])

    def __init__(self, arc: List[str]):
        self.DIRECTION = 0
        self.UTTERANCE = 1

        # Set story arc
        self.arc = arc
        self.current_arc_stage = 0
        self.complete = False

        # List of lines in the script
        self.script_lines = []
    
    # Increment arc stage
    def arc_step(self):
        # Only step if within arc length
        if self.current_arc_stage < len(self.arc) - 1:
            self.current_arc_stage += 1
        
        # Otherwise, set script to complete
        else:
            self.complete = True

    # Append list of utterances to script
    def append_dialogue(self, utterances: List[str]):
        for u in utterances:
            self.script_lines.append(
                self.ScriptLine(len(self.script_lines), self.current_arc_stage, self.UTTERANCE, u))
    
    # Append direction to script
    def append_direction(self, direction: str):
        self.script_lines.append(
            self.ScriptLine(len(self.script_lines), self.current_arc_stage, self.DIRECTION, direction))

    # Get the last n utterances from the script
    def get_prev_utterances(self, n: int):
        # ...
        count, index = 0, 0
        prev_utterances = []
        for _ in range(len(self.script_lines)):
            if count == n:
                break
            index -= 1
            line = self.script_lines[index]
            if line.type == self.DIRECTION:
                prev_utterances.insert(0, line)
                count += 1
        return prev_utterances
    
    # Get the last n directions from the script
    def get_prev_directions(self, n: int):
        # ...
        count, index = 0, 0
        prev_directions = []
        for _ in range(len(self.script_lines)):
            if count == n:
                break
            index -= 1
            line = self.script_lines[index]
            if line.type == self.DIRECTION:
                prev_directions.insert(0, line)
                count += 1
        return prev_directions

        # Get the last n directions from the script
    def get_prev_lines(self, n: int, type: int):
        # ...
        count, index = 0, 0
        prev = []
        for _ in range(len(self.script_lines)):
            if count == n:
                break
            index -= 1
            line = self.script_lines[index]
            if line.type == type:
                prev.insert(0, line)
                count += 1
        return prev
    
    # Convert script to string
    def __str__(self):
        script_str = ""
        for l in self.script_lines:
            script_str += l.text + "\n"
        return script_str

if __name__ =='__main__':
    arc = ['sad', 'happy']
    s = Script(arc)
    N = 5
    utterances = [f"utterance {i}" for i in range(N)]
    directions = [f"direction {i}" for i in range(N)]
    for u, d in zip(utterances, directions):
        s.append_dialogue([u])
        s.append_direction(d)

    # print(str(s))
    assert len(s.get_prev_utterances(n=N+1)) == len(utterances)
    assert len(s.get_prev_directions(n=0)) == 0
    assert len(s.get_prev_directions(n=3)) == 3

from collections import namedtuple

"""
Component to hold the script
"""
class Script:
    # Named tuple for a single line in the script
    ScriptLine = namedtuple('ScriptLine', ['line_num', 'arc_stage', 'type', 'text'])

    def __init__(self, arc):
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
        if self.current_arc_stage < len(arc) - 1:
            self.current_arc_stage += 1
        
        # Otherwise, set script to complete
        else:
            self.complete = True

    # Append list of utterences to script
    def append_dialogue(self, utterances):
        for u in utterances:
            script_lines.append(
                ScriptLine(len(script_lines), self.current_arc_stage, self.UTTERANCE, u))
    
    # Append direction to script
    def append_direction(self, direction):
        script_lines.append(
            ScriptLine(len(script_lines), self.current_arc_stage, self.DIRECTION, direction))

    # Get the last n utterences from the script
    def get_prev_utterences(self, n):
        # ...
        return
    
    # Get the last n directions from the script
    def get_prev_directions(self, n):
        # ...
        return
    
    # Convert script to string
    def __str__():
        script_str = ""
        for l in script_lines:
            script_str += l.text
        return script_str

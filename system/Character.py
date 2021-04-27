from collections import namedtuple
from typing import List, Tuple
from .Emotions import Emotions

"""
Container for story character
"""
class Character:
    def __init__(self, name, arc):
        self.name = name
        self.arc = arc

        self.arc_stage = 0
        self.arc_complete = False
    
    """ Get current emotion of character """
    def current_emotion(self):
        return self.arc[self.arc_stage]
    
    """ Step to next part of character arc """
    def arc_step(self):
        if self.arc_stage < len(self.arc) - 1:
            self.arc_stage += 1
        else:
            self.arc_complete = True
    
    def __str__(self):
        return self.name
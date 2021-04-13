import json

"""
Component to hold narrative frames
"""
class NarrativeFrameCollection:
    def __init__(self):
        # Load all frames from json into dictionary
        with open('narrative_frames.json') as json_file:
            self.frames = json.load(json_file)
    
    # Get all frames from collection
    def select_all(self):
        return self.frames
    
    # Maybe other methods to get a certain subset of frames
    # ...
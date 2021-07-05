import json

# NOTE: Created since we thought we may need a clean way to access only a subset of narrative frames based on some criterea

"""
Interface for accessing narrative frames
"""
class NarrativeFrameCollection:
    def __init__(self):
        # Load all frames from json into dictionary
        with open('narrative_frames.json') as json_file:
            self.frames = json.load(json_file)
    
    # Get all frames from collection
    def retrieve_all(self):
        return self.frames
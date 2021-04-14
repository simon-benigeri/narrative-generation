from .NarrativeFrameCollection import NarrativeFrameCollection
from .Script import Script

"""
Component to condition dialogue generation
"""
class ContextBuilder:
    def __init__(self):
        # ...
        pass
    
    def generate_context(self, script:Script, narrative_frame_collection:NarrativeFrameCollection):
        # Select matching narrative frame
        # ...

        # Find properties of attributes from story
        # ...

        # Do we just use those event attributes as the conditioning text?

        return "billy was holding a knife"
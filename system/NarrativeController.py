from .NarrativeFrameCollection import NarrativeFrameCollection
from .Script import Script
from .ContextBuilder import ContextBuilder
from .DialogueGenerator import DialogueGenerator
from .NarrativePlanner import NarrativePlanner

"""
Controller class to generate script using components
"""
class NarrativeController:
    def __init__(self):
        # Initialize components
        self.narrative_frame_collection = NarrativeFrameCollection()

        self.context_builder = ContextBuilder()
        self.dialogue_generator = DialogueGenerator()
        self.narrative_planner = NarrativePlanner()
    
    # Generate narrative script using components
    def generate_script(self, arc, initial_script:Script=None):
        # Initialize script if initial script not given
        if not initial_script:
            script = Script(arc=arc)
        else:
            script = initial_script

        # Generate while script is not complete
        while not script.complete:

            # Generate context
            context_text = self.context_builder.generate_context(script, self.narrative_frame_collection)

            # Generate dialogue
            utterances = self.dialogue_generator.generate_dialogue(script)
            for utterance in utterances:
                #TODO: not sure if we are appending utterances right here
                script.append_utterance(utterance[0], utterance[1])

            # Generate screen direction
            direction = self.narrative_planner.generate_direction(script)
            script.append_direction(direction)
            
            # Step to next stage in the story arc
            script.arc_step()
            
        return str(script)

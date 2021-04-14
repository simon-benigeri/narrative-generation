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
    def generate_script(self, arc):
        # Initialize script
        script = Script(arc=arc)

        # Generate while script is not complete
        while not script.complete:

            # while personX.emotion is not target_emotion or some condition to stay at this stage of the arc - needed?
            for _ in [1]: # temp loop

                # Generate dialogue
                # while (keep talking) some condition to continue dialogue in this arc stage?
                for _ in [1]: # temp loop
                    # Generate context
                    context_text = self.context_builder.generate_context(script, self.narrative_frame_collection)

                    # Generate dialogue
                    utterances = self.dialogue_generator.generate_dialogue(script)
                    for utterance in utterances:
                        script.append_utterance('?', utterance)

                # Generate screen direction
                # while (keep narrating) some condition to continue narration in this arc stage?
                for _ in [1]: # temp loop
                    direction = self.narrative_planner.generate_direction(script)
                    script.append_direction('?', direction)
            
            # Step to next stage in the story arc
            script.arc_step()
            
        return str(script)

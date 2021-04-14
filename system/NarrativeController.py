import NarrativeFrameCollection, ContextBuilder, DialogueGenerator, NarrativePlanner, Script

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
    def generate_script(self, emotion_arc):
        # Initialize script
        script = Script(arc=emotion_arc)

        # Generate while script is not complete
        while not script.complete:

            while # personX.emotion is not target_emotion or some condition to stay at this stage of the arc - needed?

                # Generate dialogue
                while # (keep talking) some condition to continue dialogue in this arc stage?
                    # Generate context
                    context_text = context_builder(script)

                    # Generate dialogue
                    dialogue = dialogue_generator.generate_utterances(script)
                    script.append_dialogue(dialogue)

                # Generate screen direction
                while # (keep narrating) some condition to continue narration in this arc stage?
                    direction = narrative_planner.generate_direction(script)
                    script.append_direction(direction)
            
            # Step to next stage in the story arc
            script.arc_step()

        return script.to_str()


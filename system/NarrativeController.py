from .NarrativeFrameCollection import NarrativeFrameCollection
from .Script import Script
from typing import List, Tuple
from .ContextBuilder import ContextBuilder
from .DialogueGenerator import DialogueGenerator
from .NarrativePlanner import NarrativePlanner
from .Character import Character
from .Emotions import Emotions

"""
Controller class to generate script using components
"""
class NarrativeController:
    def __init__(self, characters:List[Character]):
        # Set characters
        self.characters = characters

        # Initialize script if initial script not given
        self.script = Script()

        # Initialize components
        self.narrative_frame_collection = NarrativeFrameCollection()
        self.context_builder = ContextBuilder()
        self.dialogue_generator = DialogueGenerator()
        self.narrative_planner = NarrativePlanner()
    
    """ Step to next stage of each characters arc """
    def arc_step(self):
        for character in self.characters:
            character.arc_step()
    
    """ Decide on which characters should participate in dialogue and which lead/respond """
    def select_dialogue_characters(self):
        # TODO: Need to figure out how to do this or if this is even the way to go
        # NOTE: First two will always lead
        return self.characters[0], self.characters[1]
    
    """ Generate narrative script using components """
    def generate_script(self):
        # Generate script until all character arcs are complete 
        while not self.script.is_complete:
            # Generate context
            context_text = self.context_builder.generate_context(script=self.script, characters=self.characters, narrative_frame_collection=self.narrative_frame_collection)

            # Decide on which character leads the dialogue
            leading_character, response_character = self.select_dialogue_characters()

            # Generate dialogue
            utterances = self.dialogue_generator.generate_dialogue(script=self.script, context_text=context_text, leading_character=leading_character, response_character=response_character)
            for character, utterance in utterances:
                self.script.append_utterance(character=character, utterance=utterance)

            # Generate screen direction
            direction = self.narrative_planner.generate_direction(self.script)
            self.script.append_direction(direction)

            # Step to next stage of the arc
            self.arc_step()

            # Decide if story is complete
            self.script.is_complete = all([c.arc_complete for c in self.characters])
            
        return self.script

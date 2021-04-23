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
        # self.dialogue_generator = DialogueGenerator()
        #self.narrative_planner = NarrativePlanner()
    
    """ Step to next stage of each characters arc """
    def arc_step(self):
        for character in self.characters:
            character.arc_step()

    """ Return whether every characters arc is complete """
    def character_arcs_complete(self):
        return all([c.arc_complete for c in self.characters])
    
    """ Generate narrative script using components """
    def generate_script(self):
        # Generate script until all character arcs are complete TODO: this is a dumb way to loop
        while not self.character_arcs_complete():
            # TODO: some issue here wit context text
            # Generate context
            context_text = self.context_builder.generate_context(script=self.script, characters=self.characters, narrative_frame_collection=self.narrative_frame_collection)
            print("Context: ", context_text)
            return self.script

            # TODO: Need to bind characters for dialogue

            # Generate dialogue
            utterances = self.dialogue_generator.generate_dialogue(script=script, context_text=context_text)
            for character, utterance in utterances:
                #TODO: not sure if we are appending utterances right here <- what does this mean?
                self.script.append_utterance(character=character, utterance=utterance)

            # Generate screen direction
            direction = self.narrative_planner.generate_direction(script)
            self.script.append_direction(direction)

            # Step to next stage of the arc
            self.arc_step()
            
        return self.script

from .NarrativeFrameCollection import NarrativeFrameCollection
from .Script import Script
from .Character import Character
from typing import List, Tuple
import re
import random

from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

"""
Component to condition dialogue generation
"""
class ContextBuilder:
    def __init__(self):
        # ...
        #TODO: make out own model
        # the model we use is here: https://huggingface.co/deepset/roberta-base-squad2
        model_name = "deepset/roberta-base-squad2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    """ Select a narrative frame to build context around """
    def select_frame(self, script:Script, characters: List[Character], narrative_frame_collection: NarrativeFrameCollection) -> dict:
        # Gat available frames
        frames = narrative_frame_collection.retrieve_all()

        # Look through frames for emotion matches
        candidate_frames = []
        for frame in frames:
            # If current emotions match frame for both characters, break and use this frame
            if characters[0].current_emotion().name.lower() in frame["character_x_emotion"] and characters[1].current_emotion().name.lower() in frame["character_y_emotion"]:
                candidate_frames.append(frame)
                break

        # TODO: Figure out bindings

        # Select one of the candidate frames randomly
        return random.choice(candidate_frames), [(characters[0], "Character_X"),  (characters[1], "Character_Y")]
    
    """ Answer questions from the frame and add fill in event attribute text """
    def contextualize_event_attributes(self, script:Script, character_bindings:List[Tuple], frame:dict) -> List[str]:
        # Iterate over questions and fill event attribute text
        contextualized_attributes = []
        for event_attribute in frame["event_attributes"]:
            # Get answer to event attribute question given entire script
            answer = self.answer_question(question=event_attribute["question"], context=str(script))

            # Skip if no answer found
            if answer == "" or answer == "<s>":
                continue
            
            # Bind character to event attribute
            for character, character_placeholder in character_bindings:
                contextualized_attribute = re.sub(character_placeholder, character.name, event_attribute["attribute"])

            # Fill answer in event attribute text and add to list
            contextualized_attribute = re.sub("<.+>", answer, event_attribute["attribute"])
            contextualized_attributes.append(contextualized_attribute)

        return contextualized_attributes

    """ Answer a question given a context """
    def answer_question(self, question: str, context: str) -> str:
        """
        Answers 1 question given the context and the question
        Args:
            question:
            context: text containing the answer

        Returns: answer from the text
        """
        #TODO: do we add special tokens?
        inputs = self.tokenizer(question, context, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]

        # get answers
        outputs = self.model(**inputs)

        # Get the most likely beginning and end of answer with the argmax of the scores logits
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1

        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
        )

        return answer

    """ Generate context text used to condition later generators """
    def generate_context(self, script:Script, characters: List[Character], narrative_frame_collection:NarrativeFrameCollection) -> str:
        # Select matching narrative frame and bindings for characters
        frame, character_bindings = self.select_frame(script, characters, narrative_frame_collection)

        # Fill in event attributes from context
        contextualized_attributes = self.contextualize_event_attributes(script, character_bindings, frame)

        # Join centextualized events into string
        return " ".join(contextualized_attributes)

if __name__=='__main__':
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

    model_name = "deepset/roberta-base-squad2"

    # a) Get predictions
    # nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    """
    
    QA_input = {
        'question': 'Why is model conversion important?',
        'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
    }
    res = nlp(QA_input)
    print(res)
    """

    # b) Load model & tokenizer
    # model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    # model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    text = r"""
    🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
    architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
    Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
    TensorFlow 2.0 and PyTorch.
    """
    questions = [
        "How many pretrained models are available in 🤗 Transformers?",
        "What does 🤗 Transformers provide?",
        "🤗 Transformers provides interoperability between which frameworks?",
    ]
    for question in questions:
        inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]
        outputs = model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits
        answer_start = torch.argmax(
            answer_start_scores
        )  # Get the most likely beginning of answer with the argmax of the score
        answer_end = torch.argmax(
            answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        print(f"Question: {question}")
        print(f"Answer: {answer}")

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
        # Using prebuilt model, can be found here: https://huggingface.co/deepset/roberta-base-squad2
        model_name = "deepset/roberta-base-squad2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    """ Select a narrative frame to build context around """
    def select_frame(self, script:Script, characters: List[Character], narrative_frame_collection: NarrativeFrameCollection) -> (dict, List[Tuple]):
        # Get available frames
        frames = narrative_frame_collection.retrieve_all()

        # Look through frames for emotion matches
        candidate_frames = []
        for frame in frames:
            # NOTE: For now we assume character bindings will be a one-to-one map between characters and frame.characters
            character_bindings = list(zip(characters, frame["characters"]))

            # If the number of characters in the frame don't match the characters we have passed in, ignore this frame
            if len(characters) != len(frame["characters"]):
                continue

            # If any character emotions in frame don't match emotions in the arc, skip this frame
            for character, i in enumerate(characters):
                if not characters[0].current_emotion().name.lower() in frame["character_emotions"]:
                    continue

            # If frames were not skipped above, append frame and bindings as a candidate
            candidate_frames.append((frame, character_bindings))

        # Select one of the candidate frames and its bindings randomly
        random.choice(candidate_frames)
        return random.choice(candidate_frames)

    """ Answer questions from the frame and add fill in event attribute text """
    def contextualize_event_attributes(self, script:Script, character_bindings:List[Tuple], frame:dict) -> List[str]:
        # Iterate over questions and fill event attribute text
        contextualized_attributes = []
        for event_attribute in frame["event_attributes"]:
            # Bind character to event attribute and question before asking
            for character, character_placeholder in character_bindings:
                event_attribute["attribute"] = re.sub(character_placeholder, character.name, event_attribute["attribute"])
                event_attribute["question"] = re.sub(character_placeholder, character.name, event_attribute["question"])

            # Get answer to event attribute question given entire script
            answer = self.answer_question(question=event_attribute["question"], context=str(script))

            # Skip if no answer found
            if answer == "" or answer == "<s>":
                continue

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
        #TODO: do we need to add special tokens?
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

# Test
if __name__=='__main__':
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

    model_name = "deepset/roberta-base-squad2"

    # a) Get predictions
    """
    
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    QA_input = {
        'question': 'Where is Bob located?',
        'context': 'Bob is in his bedroom. Alice is in the kitchen. Alice calls Bob and asks him to come to the kitchen. Bob is in the kitchen.'
    }
    res = nlp(QA_input)
    print(res)
    """
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
    old_text = r"""
    🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
    architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
    Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
    TensorFlow 2.0 and PyTorch.
    """
    text = r"""
    Bob is in the kitchen. Bob is in the bedroom.
    """
    old_questions = [
        "How many pretrained models are available in 🤗 Transformers?",
        "What does 🤗 Transformers provide?",
        "🤗 Transformers provides interoperability between which frameworks?",
        "What architectures does hugging face provide?"
    ]
    questions = [
        "Where is Bob?"
    ]
    for question in questions:
        inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]
        outputs = model(**inputs)
        print(outputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits
        answer_start = torch.argmax(
            answer_start_scores
        )
        print(answer_start)# Get the most likely beginning of answer with the argmax of the score
        answer_end = torch.argmax(
            answer_end_scores) + 1
        print(answer_end)# Get the most likely end of answer with the argmax of the score
        words = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids))
        print(words)
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        print(f"Question: {question}")
        print(f"Answer: {answer}")

from system.NarrativeFrameCollection import NarrativeFrameCollection
from system.Script import Script
from typing import List, Tuple
import re

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

    def select_frame(self, script:Script, narrative_frame_collection: NarrativeFrameCollection) -> dict:
        # Get emotions of current arc stage
        arc_stage = script.get_current_arc_stage()

        # Gat available frames
        frames = narrative_frame_collection.retrieve_all()

        # Look through frames for emotion frames
        # TODO random selection between all valid frames
        # TODO - getting wrong one
        selected_frame = None
        for frame in frames:
            # If current emotions match frame for both characters, break and use this frame
            if arc_stage[0].name.lower() in frame["character_x_emotion"] and arc_stage[1].name.lower() in frame["character_y_emotion"]:
                selected_frame = frame
                break

        return selected_frame
    
    def answer_frame_questions(self, script:Script, frame:dict) -> List[str]:
        # Iterate over questions and fill event text
        contextualized_events = []
        for event_attribute in frame["event_attributes"]:

            # Get answer to event question given entire script
            answer = self.answer_question(question=event_attribute["question"], context=str(script))
            
            # Fill answer in event text and add to list
            contextualized_event = re.sub("<.+>", answer, event_attribute["event"])
            contextualized_events.append(contextualized_event)

        return contextualized_events

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

    
    def generate_context(self, script:Script, narrative_frame_collection:NarrativeFrameCollection) -> str:
        # Select matching narrative frame
        frame = self.select_frame(script, narrative_frame_collection)

        # Find properties of attributes from story
        contextualized_events = self.answer_frame_questions(script, frame)

        return " ".join(contextualized_events)

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

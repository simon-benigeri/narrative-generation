from system.NarrativeFrameCollection import NarrativeFrameCollection
from system.Script import Script
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

if __name__=='__main__':
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

    model_name = "deepset/roberta-base-squad2"

    # a) Get predictions
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    QA_input = {
        'question': 'Why is model conversion important?',
        'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
    }
    res = nlp(QA_input)
    print(res)

    # b) Load model & tokenizer
    # model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
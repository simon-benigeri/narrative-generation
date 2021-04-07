# Narrative Generation Proposal

Marko Sterbentz

Cameron Barrie, Harkirat Gill, Simon Benigeri

March 2021

## Outline

-   Introduction

-   System Architecture

-   Context Builder

-   Dialog Generator

-   Narrative Planner

-   Implementation Roadmap

-   References

## Introduction

Over the past few years, language models have shown incredible promise in their ability to generate syntactically coherent sentences. However, their generative capabilities are limited by relatively short attention spans and an inability to maintain the complex relationships between characters, objects, and the overall story arc. This leads to a lack of coherence in the stories that are written by these models. Despite this shortcoming, it may be possible to generate much more compelling narratives that maintain the coherence of the story by using multiple transformers trained for specific purposes and incorporating some additional structure around them. In this document, I describe a more structured, pipeline-based approach that merges the strengths of these statistical models with human knowledge about the script-writing process.

## System Architecture

The core of the system I am proposing consists of three main components: 

1. a **context builder** that uses the past story sequence and emotional arc to produce text sequences that can better condition the generation of the transformers, 
2. a **dialog generator** that uses six distinct transformers to generate dialog, and 
3. a **narrative planner** that uses a transformer to generate screen direction/action to intersperse with the dialog and help drive the plot forward. 

The context builder plays a critical role in ensuring that the dialog generator and narrative planner make use of relevant details that have already been generated as part of the script when determining what to write next. It gathers past details based on a narrative frame that matches the current emotional goals of the story. These narrative frames have associated questions that can be used to fill the slots of the frame and, once filled, can produce the text to condition the transformers with.

Further details for each component are provided below.

![](https://lh3.googleusercontent.com/2a9oBr1HvF71glhXFueuabWJvA0pWJVvo_RJbdDAEPfh-NNNzs-YYrulpL70lwIVe95TwgEt3Oyo-Hr1H3tGcixUfNPnC4uMA89N4nXSPgDHncbBFVYspYL8VNOzO0hlLxH33jac)

**System Initialization**

-   Emotional arc of the story

-   A list of discrete emotions alternating between how character X and character Y are intended to feel or respond at that point in the narrative.

-   Set of narrative frames

-   First dialog entry

-   Screen direction / action or scene description (optional)

## Context Builder

This component is responsible for producing text with which to condition the transformers in the Dialog Generator and Narrative Planner.

**Input**

-   Screen direction / action

-   The current and next emotion in the emotional arc

**Output**

-   Text with which to condition later generation

-   i.e. screen direction / action, dialog

**Mechanism**

Pick a narrative frame from the set of hand-made narrative frames that matches the expected input and output emotions for the next narrative event and dialog. For each of the event attributes in the frame, use a span retrieval transformer to find the associated property in the story text so far.

### Narrative Frames

#### Properties:

-   Input emotion (the emotion of Character X) 

-   Output emotion (how Character Y is intended to react/respond)

-   List of event attributes with associated questions for retrieval from the script text

#### Narrative Frame Example:  

**(Character X about to attack Character Y)**

```
{
    "character_x_emotion: {anger},
    "character_y_emotion": {surprise, anger, fear},
    "event_attributes:" {
        "Character_X is holding <object>": "What is Character_X holding?",
        "Character_X is <position/location>": "Where is Character_X?",
        "Character_Y is <position/location>": "Where is Character_Y?"
    }
}
```

From this, you would create a short paragraph (using the event attributes, character_X emotion, and the last two pairs of dialog) to condition the next bit of generation. For example, the above frame would be converted to the following via simple slot filling language generation:

**Text for Conditioning:**

Character_X is angry. Character_X is holding a knife. Character X is standing up. Character_Y is in front of Character_X.

Character_X: "I don't like what you've done, Character_Y".

Character_Y: "Well, I don't really like you, Character_X".

We can then predict the next bit of dialog, or the next bit of action/scene context using the Dialog Generator and Narrative Planner components.

#### Motivation

While transformers are good at producing syntactically correct sentences, they often fall down when trying to maintain coherence across long spans of text. They can quickly go off on tangents unrelated to the narrative spine of the scene and introduce new objects or characters while forgetting others. While this is detrimental to maintaining causal coherence in the story and is confusing for the human reader, it also precludes the transformer from being able to foreshadow later events since it will "forget" what it has already written.

## Dialog Generator

Building off the work done by the students in CS 338 this quarter, this component consists of 6 transformers each trained on a set of dialogs coded with one of the 6 discrete emotions in the EmotionLines dataset [1].

**Input**

-   Screen direction / action

-   The last utterance from the other character

-   The emotion of the utterance to generate

**Output**

-   The next utterances (call and response) of the characters

## Narrative Planner

This component consists of a generative transformer (e.g. GPT-2 [2]) to help drive the narrative forward by generating additional scene setup and action to go along with the dialog. This model should be fine-tuned with datasets such as RocStories [3] or GLUCOSE [4] in order to help it better generate causally coherent and meaningful actions for the script.

**Input**

-   Screen direction / action

-   The last two character utterances

**Output**

-   Screen direction / action

## Implementation Roadmap

### Phase 1: System Setup

This phase consists of the initial work to build a skeleton for the system including class definitions and functions that make accessing necessary components for generation/prediction/retrieval easy to do. The skeleton should have dummy dataflows set up and be well commented so the students know where to put their code once they have built code and models that can carry out the desired functionality. Ultimately, I think we want the focus of the undergrads to be on model building and frame generation, and not on the setup of the system skeleton.

### Phase 2: Practicum Work

The work in this phase can be done in parallel and would be where the undergrads in CS 338 can and should be heavily involved.

-   Build the Narrative Planner

    -   Fine-tune the narrative planner transformer 

        -   GPT-2 large trained on RocStories / GLUCOSE

-   Build the Dialog Generator

    -   Fine-tune the emotion transformers

        -   6 GPT-2-small trained on EmotionLines 

-   Build the Context Builder

    -   Fine-tune the span-retrieval transformer

        -   RoBERTa-large [5] trained on SQuAD 2.0 [6]

-   Build the set of narrative frames

## Phase 3: Future Work

This phase consists of work that could be done in the future, assuming that the overall approach proposed above works out. This would include exploring other decoding techniques (e.g. nucleus sampling [7]), the automatic creation of narrative frames using commonsense knowledge graphs [8], and devising better methods for deciding which narrative frame should be used next.

## References

[1] [Hsu, Chao-Chun, et al. "EmotionLines: An Emotion Corpus of Multi-Party Conversations." Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018). 2018.](https://www.aclweb.org/anthology/L18-1252.pdf)

[2][ Radford, Alec, et al. "Language models are unsupervised multitask learners." OpenAI blog 1.8 (2019): 9.](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

[3] [Mostafazadeh, Nasrin, et al. "A corpus and cloze evaluation for deeper understanding of commonsense stories." Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. 2016.](https://www.aclweb.org/anthology/N16-1098.pdf)

[4] [Mostafazadeh, Nasrin, et al. "GLUCOSE: GeneraLized and COntextualized Story Explanations." arXiv preprint arXiv:2009.07758 (2020).](https://arxiv.org/pdf/2009.07758.pdf)

[5] [Liu, Yinhan, et al. "Roberta: A robustly optimized bert pretraining approach." arXiv preprint arXiv:1907.11692 (2019).](https://arxiv.org/pdf/1907.11692.pdf)

[6] [Rajpurkar, Pranav, Robin Jia, and Percy Liang. "Know What You Don't Know: Unanswerable Questions for SQuAD." Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers). 2018.](https://www.aclweb.org/anthology/P18-2124.pdf)

[7] [Holtzman, Ari, et al. "The Curious Case of Neural Text Degeneration." International Conference on Learning Representations. 2019.](https://openreview.net/pdf/fe5e0a4c8461032e7d2c289a34236bb349b1b38a.pdf)

[8] [Sap, Maarten, et al. "Atomic: An atlas of machine commonsense for if-then reasoning." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 33. No. 01. 2019.](https://arxiv.org/pdf/1811.00146.pdf)

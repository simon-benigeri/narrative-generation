# Narrative Generation Project Documentation
## TODO:
Need to reprocess action since it is softmaxed but shouldn’t be

## Summary
### Motivation
### Objectives
### Our Approach

## Project Structure
```
- data
    - Processed
        - formatted
        - json
    - raw
- models
- pipelines
- saved_models
- system
- main.py
- narrative_frames.json
```

## System Architecture
Ideas on coherent narrative generation - what makes a coherent narrative
  - How we approached implementation - what to do with the original design from Marko
  - System outline
    - Overview - basically just - we implemented a basic architecture
    - Components - detail how they all fit in

## System Overview: 
The narrative generation system based on this design 
[](https://docs.google.com/document/d/1BEu6qF0wJBcxI6ds8qaX7O_6AXDHevwJWCr3Avpietg/edit)
components are under the system directory. Much of 

[Diagram]

## System components:
- **Narrative Controller**: The central class which manages the interaction between the components of the system in the generation loop. Calling the generate function is the starting point for generating the narrative script. How everything within the generate function works is mostly undecided so should likely be changed significantly.
- **Context Builder**: Generates a string which gathers context about what is currently happening in the story, simplified down for the dialogue generator given a narrative frame and the current script. It first finds a narrative frame which fits this scene well and then uses a span retrieval model to map the current story onto essential information about this frame and thus guides the dialogue generator into a narrative outcome inline with the narrative frame.  [Example] A basic implementation of this component using a generic pre-trained question answering model is used
- **DialogueGenerator**: Generates the next N lines of dialogue given the current script and text from the context builder. Most of our work focused on the models for this component.
- **Narrative Planner**: This component is meant to guide the arc of the story, but how it should work is mostly undecided.
- **Script**: An interface to read and write to the story script
- **Character**: Wrapper class for a character and their state in the story
- **NarrativeFrameCollection**: Interface for retrieving narrative frames. Currently only retrieves all, but may be useful if we only want to retrieve a subset of frames in the future.

In the rest of the project, we focused on...

Dialogue. Specifically, we tried to get language models to respond to a dialogue prompt with a coherent response that has a target emotion.

## Experiment Results and Next Steps

### Task

### Datasets

### Models used

### Performance metrics

### Results
1. We trained on small and medium GPT-2 models. We used data from 1 genre and then from about 20 genres. We tested on hand crafted prompts and on large samples of unseen, prompts from the script lines dataset.
    - more data improved performance
    - GPT2-medium yielded better results than GPT2-small
    - testing on extracted vs hadcrafted prompts did not affect results.

2. Tradeoff between emotion & coherence. Can we propose a test for this?

3. Improve

# Narrative Generation Project Documentation





System Architecture

Ideas on coherent narrative generation - what makes a coherent narrative
How we approached implementation - what to do with the original design from Marko?
System outline
Overview - basically just - we implemented a basic architecture
Components - detail how they all fit in

System Overview: The narrative generation system based on this design (https://docs.google.com/document/d/1BEu6qF0wJBcxI6ds8qaX7O_6AXDHevwJWCr3Avpietg/edit) components are under the system directory. Much of 

[Diagram]

System components:
Narrative Controller: The central class which manages the interaction between the components of the system in the generation loop. Calling the generate function is the starting point for generating the narrative script. How everything within the generate function works is mostly undecided so should likely be changed significantly.
Context Builder: Generates a string which gathers context about what is currently happening in the story, simplified down for the dialogue generator given a narrative frame and the current script. It first finds a narrative frame which fits this scene well and then uses a span retrieval model to map the current story onto essential information about this frame and thus guides the dialogue generator into a narrative outcome inline with the narrative frame.  [Example] A basic implementation of this component using a generic pre-trained question answering model is used
DialogueGenerator: Generates the next N lines of dialogue given the current script and text from the context builder. Most of our work focused on the models for this component.
Narrative Planner: This component is meant to guide the arc of the story, but how it should work is mostly undecided.
Script: An interface to read and write to the story script
Character: Wrapper class for a character and their state in the story
NarrativeFrameCollection: Interface for retrieving narrative frames. Currently only retrieves all, but may be useful if we only want to retrieve a subset of frames in the future.

In the rest of the project, we focused on...

Movie Script Data

Overview: To train a controlled dialogue generation model, we needed data which captured the range of possible emotions, intentions, causal dynamics which we wanted to generate. For this reason, we decided not to use emotion tagged data line EmotionLines as the text from this dataset was quite different from the distribution we wanted to generate. We instead used scripts from the Internet Movie Script Database (IMSDb) (www.imsdb.com), which has thousands of movie scripts constantly being added in 22 genres, and auto-tagged the emotions on each dialogue using a pre-trained emotion detection model. The goal was then to experiment with various formats of the data to then train the model to learn to generate dialogue and control for emotion.

Processing: Due to the irregular format of movie scripts, processing the text into a more structured and usable format was tricky. We used this github repository (https://github.com/drwiner/ScreenPy) which had processed 2564 raw movie scripts from IMSDb into JSON files labeling each line as a screen direction or dialogue along with the character associated with the dialogue. Still, there were a number of errors in the processed JSONs from this repository, the most common including mislabeling of stage direction and dialogue, misattribution of a character to an utterance, grouping together of stage direction and dialogue into one line, and some parsed json files being completely empty. We removed all script JSONs which were empty and didn’t have any labeled stage directions or dialogue, but many less obvious mistakes like those mentioned remained. This reduced the number of movie scripts for training from 2564 to ???. We decided to move forward with this data regardless of these errors as we believed the effect on the model training would be minor. This original data is under the data/raw/ScreenPyOutput directory.

Emotion Tagging: We next tagged each line of type dialogue with an emotion predicted by a pre-trained emotion classification model (https://huggingface.co/mrm8488/t5-base-finetuned-emotion). The model gave scores for each line of dialogue as being classified as joy, sadness, anger, fear, surprise, or love. Since emotion tagging took quite some time, the JSON for each script was saved in a modified format with the emotion label under data/processed/json. Instead of labeling a text with the emotion of the highest score, we tagged it with each score of each emotion so we could then experiment with different thresholds for a neutral label.

Format: Saving the emotion labeled JSON data allowed us to experiment with various formats to feed into the model for training quickly without having to run the emotion tagging every time. The simplest format to feed into the model was as a call-response with just a pair of sequential utterances. For example…

C: Okay, I’ll tell you. Do you know the muffin man?
R: The muffin man?

There were many ways we could have gone about conditioning the models to generate text of a specific emotion. One would be to train a single large model with this format, with emotion tags added in…

C (fear): Okay, I’ll tell you. Do you know the muffin man?
R (neutral): The muffin man?

Another we tried was to train a small model for each emotion and train it only on the data for where the response was tagged with that emotion. For example, a model trained on joy would be given the following format...

C: How did the game go?
R: We won!

Eventually, the goal is not just to include emotional information, but also information about the story context, the intent or goal of each character to better capture the intended trajectory of the story and make it more coherent.

Proposed Formats:

<dialogue-start>

	<context-start>Bob is watching TV. It is midnight.<context-end>

	<characters-start>
		Bob: (Emotion: Sad, Goal: Get mind off work, Outcome: Unable to get mind off work)
		Alice: (Emotion: Neutral, Goal: Figure out what happened, Outcome: Learn what happened)
	<characters-end>

	<utterances-start>
		Alice: Are you okay? You’ve been pretty quiet.
		Bob: Yeah I’m fine.
<utterances-end>
<dialogue-end>

In this case, special tokens for each start and end tag need to be added into the tokenizer. This is just a proposal and can likely be simplified to make training for the model easier.

Creating these formats from the emotion tagged JSON to train the model take only a couple seconds to run and are saved under data/processed/formatted.

Emotion Control Experiment:
Emotion specific transformers
All emotion transformers
Evaluation
What did we actually do other than this?

Observations/Findings
Conditioning both all and emotion specific models with emotions works
Tradeoff between emotion control and coherence - maybe


Ideas for Next Steps
Compare all emotion model effectiveness with emotion specific
Try cleaner data
Add context - more dialogue or use directions
Try many emotions
Train like a translation task rather than a language model task (idk about GPT-2)


Core Questions
Questions on how to implement the actual system
Need this section or just merge into next steps?



Resources
Narrative Structure
Hero goal sequence: https://www.thestorysolution.com/wp-content/uploads/2010/07/FINDINGNEMOHGS.pdf


Movie script data
IMSDb: 
IMSDb ScreenPy JSON: 
Cornell Dialogues Corpus: https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
Multi turn dialogue: https://arxiv.org/pdf/1710.03957.pdf
https://parl.ai/about/
Emotion text classification
Model used for emotion tagging: https://huggingface.co/mrm8488/t5-base-finetuned-emotion
CARER: Contextualized Affect Representations for Emotion Recognition
Prompt formatting
https://arxiv.org/pdf/2101.00190.pdf
Useful tutorials and guides
Generate text using hugging face transformers: https://huggingface.co/blog/how-to-generate
Related projects
Planning based
https://www.semanticscholar.org/paper/MEXICA%3A-A-computer-model-of-a-cognitive-account-of-P%C3%A9rez-Sharples/ea275de9890489706bc21b533b44843b134bad4d
https://www.semanticscholar.org/paper/Generating-Story-Analogues-Riedl-Le%C3%B3n/8645dfbc3378b3445ff50d4034099d7b92ce3c81
https://www.semanticscholar.org/paper/An-Offline-Planning-Approach-to-Game-Plotline-Li-Riedl/b19dace598d7ccbf9a0a4df3fece08372d874642
https://arxiv.org/abs/1811.05701
Character based
https://www.semanticscholar.org/paper/Characters-in-Search-of-an-Author%3A-AI-Based-Virtual-Cavazza-Charles/8fa7b8c47f38b2c9e683f916a03002995e42f9b3
https://www.semanticscholar.org/paper/Building-Synthetic-Actors-for-Interactive-Dramas-Louchart-Aylett/9ea91a4e070f3bfdf84f809ec40f3c9d3e748916
Neural
https://arxiv.org/abs/1706.01331
https://arxiv.org/abs/1805.04833


Other
https://arxiv.org/pdf/2002.02878.pdf
https://research.fb.com/wp-content/uploads/2019/11/Learning-to-Speak-and-Act-in-a-Fantasy-Text-Adventure-Game.pdf
https://parl.ai/projects/light/


import os
import time
import datetime

import pandas as pd
import seaborn as sns
import numpy as np
import random
import re

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
torch.manual_seed(42)

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup

import nltk
nltk.download('punkt')


# Config
GPT_MODEL = os.environ.get('GPT_MODEL', 'gpt2')

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 2))
EPOCHS = int(os.environ.get('EPOCHS', 1))
LEARNING_RATE = 2e-5
WARMUP_STEPS = 1e2
EPSILON = 1e-8
NUM_TRAIN_SCRIPTS = int(os.environ.get('NUM_TRAIN_SCRIPTS', 0)) # 0 means read all

SAMPLE_EVERY = 0

SEED = 37

# Config
GENRE = os.environ.get('GENRE', 'Action')
EMOTIONS = os.environ.get('EMOTIONS', 'emotions')
genre_extension = f'{GENRE}/' if EMOTIONS in ['emotions', 'no_emotions'] else ''
READ_SCRIPTS_DIR = f'../data/processed/formatted/{EMOTIONS}/{genre_extension}'
SAVE_MODEL_DIR = f'../models/{GPT_MODEL}/{EMOTIONS}/{GENRE}/{EPOCHS}_epochs/model_save/'

""" Prepare Dataset for GPT2 """
class GPT2Dataset(Dataset):
    def __init__(self, txt_list, tokenizer, gpt2_type=GPT_MODEL, max_length=768):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []

        for txt in txt_list:
            encodings_dict = tokenizer('<|startoftext|>'+ txt + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

""" Get list of scenes formatted for training """
def get_formatted_script_scenes(scripts_dir, num_scripts=0):
    scenes = []
    for i, filename in enumerate(os.listdir(scripts_dir)):
        if num_scripts > 0 and i > num_scripts:
            break

        with open(os.path.join(scripts_dir, filename)) as f:
            script_scenes = "\n".join(f.readlines()).split("<new-scene>")
            for s in script_scenes:
                script_lines = s.split("\n")
                lines=""
                for line in script_lines:
                    if line.startswith("<start-line>"):
                        lines += re.sub("\<.*?\>", "", line)
                if lines:
                    scenes.append(lines)
    return scenes

""" Get list of formatted call/responses for training """
def get_formatted_call_responses(scripts_dir, num_scripts=0):
    lines = []
    for i, filename in enumerate(os.listdir(scripts_dir)):
        if num_scripts > 0 and i >= num_scripts:
            break

        with open(os.path.join(scripts_dir, filename)) as f:
            lines += "".join(f.readlines()).split("---")
    return lines


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

""" Train model """
def train(model, epochs, train_dataloader, validation_dataloader):
    optimizer = AdamW(model.parameters(), lr = LEARNING_RATE, eps = EPSILON) # (AdamW from huggingface not pytorch)
    total_steps = len(train_dataloader) * EPOCHS # (Not same as training samples)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = WARMUP_STEPS, num_training_steps = total_steps) # Scheduler changes the learning rate as the training loop progresses

    total_t0 = time.time()
    training_stats = []

    for epoch_i in range(0, epochs):
        print('\n======== Epoch {:} / {:} ========\n'.format(epoch_i + 1, epochs))
        t0 = time.time()
        total_train_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(model.device)
            b_labels = batch[0].to(model.device)
            b_masks = batch[1].to(model.device)

            model.zero_grad()        
            outputs = model(b_input_ids,
                            labels=b_labels, 
                            attention_mask = b_masks,
                            token_type_ids=None
                        )

            loss = outputs[0]  

            batch_loss = loss.item()
            total_train_loss += batch_loss

            # Get sample every x batches.
            if SAMPLE_EVERY != 0 and step % sample_every == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), batch_loss, elapsed))

                # Sample
                model.eval()
                sample_outputs = model.generate(
                                        bos_token_id=random.randint(1,30000),
                                        do_sample=True,   
                                        top_k=50, 
                                        max_length = 200,
                                        top_p=0.95, 
                                        num_return_sequences=1
                                    )             
                for i, sample_output in enumerate(sample_outputs):
                    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
                
                model.train()

            loss.backward()
            optimizer.step()
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)       
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("\nAverage training loss: {0:.2f}".format(avg_train_loss))
        print("\nTraining epoch took: {:}".format(training_time))

        t0 = time.time()
        model.eval()
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(model.device)
            b_labels = batch[0].to(model.device)
            b_masks = batch[1].to(model.device)
            
            with torch.no_grad():        
                outputs  = model(b_input_ids,            
                                #token_type_ids=None, 
                                attention_mask = b_masks,
                                labels=b_labels)
            
                loss = outputs[0]

            batch_loss = loss.item()
            total_eval_loss += batch_loss        

        avg_val_loss = total_eval_loss / len(validation_dataloader)
        validation_time = format_time(time.time() - t0)    

        print("Validation Loss: {0:.2f}".format(avg_val_loss))
        print("Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )
    
    return model, training_stats

""" Plot training and validation curve """
def plot_training_stats(training_stats):
    pd.set_option('precision', 2)
    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index('epoch')

    # A hack to force the column headers to wrap.
    #df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

    # Display the table.
    #df_stats

    sns.set(style='darkgrid')
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)

    # Plot the learning curve.
    plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
    plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

    # Label the plot.
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks(range(1, EPOCHS))
    plt.savefig(f"{SAVE_MODEL_DIR}train.png")
    plt.show()

""" Generate some sample outputs from the transformer """
def generate_samples(model, prompt):
    model.eval()
    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(model.device)

    sample_outputs = model.generate(
                                generated, 
                                #bos_token_id=random.randint(1,30000),
                                do_sample=True,   
                                top_k=50, 
                                max_length = 300,
                                top_p=0.90, 
                                num_return_sequences=3
                            )

    return [tokenizer.decode(sample_output, skip_special_tokens=True) for sample_output in sample_outputs]

""" Save model """
def save_model(model, output_dir):
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`. They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Load the GPT tokenizer.
    tokenizer = GPT2Tokenizer.from_pretrained(GPT_MODEL, bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>', additional_special_tokens=["<start-line>", "<end-line>"])

    # Inititalize dataset
    print("Preparing data...")
    lines = get_formatted_call_responses(READ_SCRIPTS_DIR, NUM_TRAIN_SCRIPTS)
    dataset = GPT2Dataset(lines, tokenizer, max_length=256)

    # Split into training and validation sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))
    print('{:>5,} test samples'.format(len(dataset)-train_size-val_size))

    # Create DataLoaders
    train_dataloader = DataLoader(
                train_dataset,
                sampler = RandomSampler(train_dataset), # Shuffle
                batch_size = BATCH_SIZE
            )

    validation_dataloader = DataLoader(
                val_dataset,
                sampler = SequentialSampler(val_dataset), # Order doesn't matter
                batch_size = BATCH_SIZE
            )

    # Initialize model
    print("Initializing model...")
    configuration = GPT2Config.from_pretrained(GPT_MODEL, output_hidden_states=False) # Not used here. Why are we using it? I don't know. A riddle for future generations
    model = GPT2LMHeadModel.from_pretrained(GPT_MODEL, config=configuration)
    model.resize_token_embeddings(len(tokenizer)) # Match model embeddings to tokenizer embeddings
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        model.cuda()
    else:
        print("No GPU available")

    # Train
    print("Training...")
    model, training_stats = train(model, EPOCHS, train_dataloader, validation_dataloader)

    # Print training stats
    print("Training stats...")
    
    # Save model
    save_model(model, SAVE_MODEL_DIR)
    plot_training_stats(training_stats)
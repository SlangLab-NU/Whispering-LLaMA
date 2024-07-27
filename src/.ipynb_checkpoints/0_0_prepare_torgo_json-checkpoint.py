#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
import sys

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve() # does not work as jupyter notebook 
sys.path.append(str(wd))

# This notebook uses https://github.com/openai/whisper with edits to the whisper_openAI/decoding.py to generate multiple hypothesis
import sys
import os
import argparse
import re
import json
import torch
import logging
import pandas as pd
import numpy as np

from dotenv import load_dotenv
from datasets import load_dataset, DatasetDict, Audio
from dataclasses import dataclass
from typing import Dict, List, Union
from evaluate import load
from tqdm import tqdm
from datetime import datetime
import time



start_time = time.time() 

parser = argparse.ArgumentParser(
    description='Process speaker ID and optional parameters.')

# Required argument: speaker ID
parser.add_argument('--speaker_id',
                    type=str,
                    help='Speaker ID in the format [MF]C?[0-9]{2}')
# Optional arguments for training parameters
parser.add_argument('--learning_rate', type=float,
                    default=0.0001, help='Learning rate (default: 0.0001)')
parser.add_argument('--train_batch_size', type=int,
                    default=4, help='Training batch size (default: 4)')
parser.add_argument('--eval_batch_size', type=int,
                    default=4, help='Evaluation batch size (default: 4)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed (default: 42)')
parser.add_argument('--gradient_accumulation_steps', type=int,
                    default=2, help='Gradient accumulation steps (default: 2)')
parser.add_argument('--optimizer', type=str, default='adamw_torch',
                    help='Optimizer type (default: adamw_torch)')
parser.add_argument('--lr_scheduler_type', type=str, default='linear',
                    help='Learning rate scheduler type (default: linear)')
parser.add_argument('--num_epochs', type=int, default=20,
                    help='Number of epochs (default: 20)')

# Other optional arguments
parser.add_argument('--keep_all_data', action='store_true',
                    help='Keep all data in the test set')
parser.add_argument('--debug', action='store_true',
                    help='Enable debug mode (default: False)')
parser.add_argument('--repo_suffix', type=str,
                    default='', help='Repository suffix')

parser.add_argument('--model_name', type=str,
                    default='tiny', help='Whisper model to load (default: tiny.en)')
parser.add_argument('--best_of', type=int, default=200, help='Number of best sequences to choose from during decoding')


args = parser.parse_args()

    
speaker_id = args.speaker_id
test_speaker = args.speaker_id
model_name = args.model_name
best_of = args.best_of

learning_rate = 0.0001
train_batch_size = 4
eval_batch_size = 4
seed = 42
gradient_accumulation_steps = 2
optimizer = "adamw_torch"
lr_scheduler_type = "linear"
num_epochs = 20
keep_all_data = False
debug = False
repo_suffix = ""

if args.repo_suffix and not re.match(r'^[_-]', args.repo_suffix):
    repo_suffix = '_' + args.repo_suffix

print(f"Speaker ID: {speaker_id}")
print(f"Learning rate: {learning_rate}")
print(f"Training batch size: {train_batch_size}")
print(f"Evaluation batch size: {eval_batch_size}")
print(f"Random seed: {seed}")
print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
print(f"Optimizer type: {optimizer}")
print(f"Learning rate scheduler type: {lr_scheduler_type}")
print(f"Number of epochs: {num_epochs}")
print(f"Keep all data: {keep_all_data}")
print(f"Debug mode: {debug}")
print(f"Repository suffix: {repo_suffix}")
print(f"model_name: {model_name}")


test_speaker = speaker_id

if repo_suffix and not re.match(r'^[_-]', args.repo_suffix):
    repo_suffix = '_' + repo_suffix


# In[3]:


import os
import logging

# Define the path to the CSV file
torgo_csv_path = "data_preparation/torgo.csv"

# Check if the path exists and is a file
if os.path.exists(torgo_csv_path) and os.path.isfile(torgo_csv_path):
    print("The CSV file exists.")
else:
    print("The CSV file does not exist.")

torgo_dataset_path = '/work/van-speech-nlp/data/torgo'
torgo_dataset_dir_path = torgo_dataset_path + \
        '/' if torgo_dataset_path[-1] != '/' else torgo_dataset_path
output_path = 'output'
print(f'torgo_dataset_path: {torgo_dataset_path}')
print(f'torgo_dataset_dir_path: {torgo_dataset_dir_path}')

repo_name = f'torgo_tiny_finetune_{test_speaker}{repo_suffix}'
repo_path = f'jindaxz/{repo_name}'

# Path to save model / checkpoints{repo_name}'
model_local_path = output_path + '/model/' + repo_name



# In[5]:


import numpy
# Renamed the Whisepr repo (https://github.com/openai/whisper) with the changed decoding.py file as whisper_openAI
import whisper_openAI.whisper as whisper
import torch
import tqdm
model, _ = whisper.load_model(f"{model_name}") # you can change the whisper model here to largev2 or large to swap the  model.


# In[6]:


data_df = pd.read_csv(torgo_csv_path)
dataset_csv = load_dataset('csv', data_files=torgo_csv_path)

# Check if the following columns exist in the dataset ['session', 'audio', 'text', 'speaker_id']
expected_columns = ['session', 'audio', 'text', 'speaker_id']
not_found_columns = []
for column in expected_columns:
    if column not in dataset_csv['train'].column_names:
        not_found_columns.append(column)

if len(not_found_columns) > 0:
    logging.error(
        "The following columns are not found in the dataset:" + " [" + ", ".join(not_found_columns) + "]")
    sys.exit(1)


# In[7]:


logging.info(
    "Splitting the dataset into training / validation / test sets...")

# Extract the unique speakers in the dataset
speakers = data_df['speaker_id'].unique()

logging.info("Unique speakers found in the dataset:")
logging.info(str(speakers) + '\n')

if test_speaker not in speakers:
    logging.error("Test Speaker not found in the dataset.")
    sys.exit(1)

valid_speaker = 'F03' if test_speaker != 'F03' else 'F04'
train_speaker = [s for s in speakers if s not in [
    test_speaker, valid_speaker]]

torgo_dataset = DatasetDict()
torgo_dataset['train'] = dataset_csv['train'].filter(
    lambda x: x in train_speaker, input_columns=['speaker_id'])
torgo_dataset['validation'] = dataset_csv['train'].filter(
    lambda x: x == valid_speaker, input_columns=['speaker_id'])
torgo_dataset['test'] = dataset_csv['train'].filter(
    lambda x: x == test_speaker, input_columns=['speaker_id'])


# In[8]:


original_data_count = {'train': len(torgo_dataset['train']), 'validation': len(
    torgo_dataset['validation']), 'test': len(torgo_dataset['test'])}

if not keep_all_data:
    # Update the three dataset splits (if ['test_data'] == 1, keep in test, if ['test_data'] == 0, keep in train and validation)
    torgo_dataset['train'] = torgo_dataset['train'].filter(
        lambda x: x['test_data'] == 0)
    torgo_dataset['validation'] = torgo_dataset['validation'].filter(
        lambda x: x['test_data'] == 0)
    torgo_dataset['test'] = torgo_dataset['test'].filter(
        lambda x: x['test_data'] == 1)

    # Drop the 'test_data' column
    torgo_dataset['train'] = torgo_dataset['train'].remove_columns([
                                                                   'test_data'])
    torgo_dataset['validation'] = torgo_dataset['validation'].remove_columns([
                                                                             'test_data'])
    torgo_dataset['test'] = torgo_dataset['test'].remove_columns([
                                                                 'test_data'])
    logging.info(
        f"After removal of repeated prompts, the number of data in each dataset is:")
    logging.info(
        f'Train:       {len(torgo_dataset["train"])}/{original_data_count["train"]} ({len(torgo_dataset["train"]) * 100 // original_data_count["train"]}%)')
    logging.info(
        f'Validation:  {len(torgo_dataset["validation"])}/{original_data_count["validation"]} ({len(torgo_dataset["validation"]) * 100 // original_data_count["validation"]}%)')
    logging.info(
        f'Test:        {len(torgo_dataset["test"])}/{original_data_count["test"]} ({len(torgo_dataset["test"]) * 100 // original_data_count["test"]}%)\n')


# In[9]:


# Remove special characters from the text
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\`\�0-9]'


def remove_special_characters(batch):
    batch['text'] = re.sub(chars_to_ignore_regex,
                           ' ', batch['text']).lower()
    return batch

torgo_dataset = torgo_dataset.map(remove_special_characters)


# In[ ]:


print(torgo_dataset['train'][0]['text'])


# In[10]:


## convert the sample rate of every audio files using cast_column function
torgo_dataset = torgo_dataset.cast_column("audio", Audio(sampling_rate=16000))


# In[11]:


# Define the minimum and maximum input length in seconds
min_input_length_in_sec = 1.0
max_input_length_in_sec = 10.0
sampling_rate=16000

# Define the filtering functions based on input length
def filter_min_length(example):
    return example["audio"]["array"].shape[0] > min_input_length_in_sec * sampling_rate

def filter_max_length(example):
    return example["audio"]["array"].shape[0] < max_input_length_in_sec * sampling_rate

# Apply the filters
torgo_dataset = torgo_dataset.filter(filter_max_length)
torgo_dataset = torgo_dataset.filter(filter_min_length)


# In[12]:


logging.info(
    f"After filter, the number of data in each dataset is:")
logging.info(
    f'Train:       {len(torgo_dataset["train"])}/{original_data_count["train"]} ({len(torgo_dataset["train"]) * 100 // original_data_count["train"]}%)')
logging.info(
    f'Validation:  {len(torgo_dataset["validation"])}/{original_data_count["validation"]} ({len(torgo_dataset["validation"]) * 100 // original_data_count["validation"]}%)')
logging.info(
    f'Test:        {len(torgo_dataset["test"])}/{original_data_count["test"]} ({len(torgo_dataset["test"]) * 100 // original_data_count["test"]}%)\n')



train_dataset = torgo_dataset["train"]
validation_dataset = torgo_dataset["validation"]
test_dataset = torgo_dataset["test"]


import json
import os
import tqdm
import numpy as np


# Function to process a single item
def process_item(item, best_of, model):
    audio = item['audio']['array'].astype(np.single)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    ground_truth = item['text'].replace(' <COMMA>', ',').replace(' <PERIOD>', '.').replace(' <QUESTIONMARK>', '?').replace(' <EXCLAMATIONPOINT>', '!').lower()
    source = 'NP-Torgo'
    cat = 'NP-Torgo'
    time_length = len(audio) / 16000
    path_to_file = item['audio']['path']
    random_temperature = np.random.randint(70, 81) / 100
    options = whisper.DecodingOptions(fp16=True, without_timestamps=True, temperature=random_temperature, best_of=best_of, language='english')
    result, _ = whisper.decode(model, mel, options)
    result = list(result)

    if len(result) <= 10:
        if random_temperature < 0.75:
            random_temperature += 0.2
        else:
            random_temperature += 0.1
        options = whisper.DecodingOptions(fp16=True, without_timestamps=True, temperature=random_temperature, best_of=best_of, language='english')
        result, _ = whisper.decode(model, mel, options)
        result = list(result)

    return {
        item['session']: {
            'temp': random_temperature,
            'path': path_to_file,
            'ground_truth': ground_truth,
            'inference': result,
            'source': source,
            'category': cat,
            'time': time_length,
            'path': path_to_file
        }
    }

# Function to generate inference JSON with checkpointing
def generate_inference_json(dataset, dataset_name, checkpoint_interval=10):
    save_path = f'Inference/gs_inferences/{str(dataset_name)}.json'
    checkpoint_path = f'Inference/json_checkpoint/{str(dataset_name)}_checkpoint.json'

    checkpoint_dir = os.path.dirname(checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Created checkpoint directory: {checkpoint_dir}")

    # Load progress from checkpoint if exists
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as file:
            to_json = json.load(file)
        start_index = len(to_json)
    else:
        to_json = []
        start_index = 0

    for i in tqdm.tqdm(range(start_index, len(dataset))):
        item = dataset[i]
        result = process_item(item, best_of, model)
        to_json.append(result)

        # Save checkpoint every `checkpoint_interval` items
        if (i + 1) % checkpoint_interval == 0:
            with open(checkpoint_path, "w") as file:
                json.dump(to_json, file, indent=4)

    # Save final output
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as file:
        json.dump(to_json, file, indent=4)

    # Remove checkpoint file after completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


# saved dir is in Inference/gs_inferences
generate_inference_json(train_dataset, f'torgo_train_{speaker_id}_{model_name}')
generate_inference_json(validation_dataset, f'torgo_val_{speaker_id}_{model_name}')
generate_inference_json(test_dataset, f'torgo_test_{speaker_id}_{model_name}')

end_time = time.time()

elapsed_time = end_time - start_time

elapsed_time_minutes = elapsed_time / 60

print(f"script runtime {elapsed_time_minutes:.2f}")
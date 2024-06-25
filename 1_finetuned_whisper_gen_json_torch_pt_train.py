#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


args = parser.parse_args()

# Check if the speaker ID is valid
if not re.match(r'^[MF]C?[0-9]{2}$', args.speaker_id):
    print("Please provide a valid speaker ID.")
    sys.exit(1)
    
speaker_id = args.speaker_id
test_speaker = args.speaker_id
model_name = args.model_name

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

if not re.match(r'^[MF]C?[0-9]{2}$', speaker_id):
    print("Please provide a valid speaker ID.")
    sys.exit(1)
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

pretrained_model_name = "openai/whisper-tiny"


import os
MODEL_PATH = f"finetuned_whisper_output/model/torgo_tiny_finetune_{speaker_id}_frozen_encoder/pytorch_model.bin"

if os.path.exists(MODEL_PATH):
    print("The file exists.")
else:
    print("The file does not exist.")


import re
import whisper_openAI.whisper as whisper

# https://github.com/openai/whisper/discussions/830
def hf_to_whisper_states(text):
    text = re.sub('.layers.', '.blocks.', text)
    text = re.sub('.self_attn.', '.attn.', text)
    text = re.sub('.q_proj.', '.query.', text)
    text = re.sub('.k_proj.', '.key.', text)
    text = re.sub('.v_proj.', '.value.', text)
    text = re.sub('.out_proj.', '.out.', text)
    text = re.sub('.fc1.', '.mlp.0.', text)
    text = re.sub('.fc2.', '.mlp.2.', text)
    text = re.sub('.fc3.', '.mlp.3.', text)
    text = re.sub('.fc3.', '.mlp.3.', text)
    text = re.sub('.encoder_attn.', '.cross_attn.', text)
    text = re.sub('.cross_attn.ln.', '.cross_attn_ln.', text)
    text = re.sub('.embed_positions.weight', '.positional_embedding', text)
    text = re.sub('.embed_tokens.', '.token_embedding.', text)
    text = re.sub('model.', '', text)
    text = re.sub('attn.layer_norm.', 'attn_ln.', text)
    text = re.sub('.final_layer_norm.', '.mlp_ln.', text)
    text = re.sub('encoder.layer_norm.', 'encoder.ln_post.', text)
    text = re.sub('decoder.layer_norm.', 'decoder.ln.', text)
    text = re.sub('proj_out.weight', 'decoder.token_embedding.weight', text)
    return text

# Load HF Model
hf_state_dict = torch.load(MODEL_PATH)    # pytorch_model.bin file
# print(hf_state_dict)

# Rename layers
for key in list(hf_state_dict.keys())[:]:
    new_key = hf_to_whisper_states(key)
    hf_state_dict[new_key] = hf_state_dict.pop(key)

# Init Whisper Model and replace model weights
model,_ = whisper.load_model('tiny')
model.load_state_dict(hf_state_dict)


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


def generate_inference_json(dataset, dataset_name):
    to_json = []
    for i, item in enumerate(tqdm.tqdm(dataset)):
        # print(item)
        audio = item['audio']['array'].astype(np.single)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        ground_truth = item['text'].replace(' <COMMA>', ',').replace(' <PERIOD>', '.').replace(' <QUESTIONMARK>', '?').replace(' <EXCLAMATIONPOINT>', '!').lower()
        source = 'NP-Torgo'
        cat = 'NP-Torgo'
        time = len(audio)/16000
        path_to_file = item['audio']['path']
        random_temperature = np.random.randint(70, 81) / 100
        options = whisper.DecodingOptions(fp16=True, without_timestamps=True, temperature=random_temperature, best_of=200, language='english')
        result, _ = whisper.decode(model, mel, options)
        result = list(result)

        if len(result) <= 10:
            if random_temperature < 0.75:
                random_temperature += 0.2
            else:
                random_temperature += 0.1
            options = whisper.DecodingOptions(fp16=True, without_timestamps=True, temperature=random_temperature, best_of=200)
            result, _ = whisper.decode(model, mel, options)
            result = list(result)

        to_json.append({
            item['session']: {
                'temp': random_temperature,
                'path': path_to_file,
                'ground_truth': ground_truth,
                'inference': result,
                'source': source,
                'category': cat,
                'time': time,
                'path': path_to_file
            }
        })

    os.makedirs(f"Inference/gs_inferences", exist_ok=True)
    save_path = f'Inference/gs_inferences/{str(dataset_name)}.json'
    with open(save_path, "w") as file:
        json.dump(to_json, file, indent=4)


# saved dir is in Inference/gs_inferences
generate_inference_json(train_dataset, f'finetuned_torgo_train_{speaker_id}')
generate_inference_json(validation_dataset, f'finetuned_torgo_val_{speaker_id}')
generate_inference_json(test_dataset, f'finetuned_torgo_test_{speaker_id}')





import os
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import whisper_openAI.whisper as whisper
import torch
from whisper_openAI.whisper.tokenizer import Tokenizer, get_tokenizer
import torch
import torch.nn.functional as F
from torch import Tensor

# We get the acoustic embeddings from Whisper Large V2
model,processor = whisper.load_model("large-v2")
# model,processor = whisper.load_model("medium")


model.eval()

import json

# The below is the json file you can generate using the "To generatn-best hyporhesis.ipynb" notebook; Need to further tokenize the hypothesis

with open(f'Inference/gs_inferences/finetuned_torgo_train_{speaker_id}.json', "r") as file:  # Change the file path and name here
    train_data = json.load(file)

with open(f'Inference/gs_inferences/finetuned_torgo_val_{speaker_id}.json', "r") as valid_file:
    val_data = json.load(valid_file)

# Load the test set
with open(f'Inference/gs_inferences/finetuned_torgo_test_{speaker_id}.json', "r") as test_file:
    test_data = json.load(test_file)

"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import sys
from pathlib import Path
import torch
import requests
import json
import os 

from lit_llama.tokenizer import Tokenizer
from tqdm import tqdm

tokenizer_path: Path = Path("weights/tokenizer.model")
tokenizer = Tokenizer(tokenizer_path)
print(f"train has {len(train_data):,} samples")
print("Processing train split ...")


def tokenize(tokenizer: Tokenizer, string: str, max_length: int, eos=True) -> torch.Tensor:
    return tokenizer.encode(string, bos=True, eos=eos, max_length=max_length)
    
def process_train_data(train_data):
    instruction = 'You are an ASR transcript selector. You have a few transcripts generated by an automatic speech recognition model. Your task is to generate the most likely transcript from them. If the generated transcripts have grammatical or logical errors, you will modify them accordingly to produce the most accurate and coherent transcript.'
    result = []

    for i in tqdm(range(len(train_data))):        
        for name in train_data[i].keys():
            ip = train_data[i][name]
        inference = ip['inference']
        gt = ip['ground_truth']
            
        # Removing the ground_truth, if present among the inferences for the prompt
        if gt in inference:
            inference.remove(gt)
                
        # Joining the inputs with '\n'
        for_input = '\n'.join(inference[:15])
        # The prompt follows the Alpaca template
        full_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{for_input}\n\n### Response:"""
        full_prompt_and_response = full_prompt + gt

        encoded_full_prompt = tokenize(tokenizer, full_prompt, max_length=2048, eos=False)
        encoded_full_prompt_and_response = tokenize(tokenizer, full_prompt_and_response, eos=True, max_length=2048)
        labels = encoded_full_prompt_and_response.clone()
        labels_with_masked_input = encoded_full_prompt_and_response.clone()
        labels_with_masked_input[:len(encoded_full_prompt)] = -1
        
        path = ip['path']
        audio = whisper.load_audio(path)  
        audio = whisper.pad_or_trim(audio)            
        mel = whisper.log_mel_spectrogram(audio).to(model.device)  # Adjust as needed for your model
        mel = mel.unsqueeze(0)
        
        with torch.no_grad():
            audio_features = model.encoder(mel)
        
        result.append({**ip, 'index': name, "input_ids": encoded_full_prompt_and_response, "input_ids_no_response": encoded_full_prompt, "labels": labels, 'labels_with_masked_input': labels_with_masked_input, 'audio_features': audio_features.bfloat16()})

    return result



split = "train"
result = process_train_data(train_data)
torch.save(result,f'Inference/gs_inferences/finetuned_torgo_{speaker_id}_{split}.pt')

split = "val"
result = process_train_data(validation_dataset)
torch.save(result,f'Inference/gs_inferences/finetuned_torgo_{speaker_id}_{split}.pt')

split = "test"
result = process_train_data(test_data)
torch.save(result,f'Inference/gs_inferences/finetuned_torgo_{speaker_id}_{split}.pt')
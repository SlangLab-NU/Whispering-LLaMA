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


# In[2]:

speakers_to_process = [
    'F01', 
    'F03', 
    'F04', 
    'M01', 
    'M02', 
    'M03', 
    'M04', 
    'M05'
]


for speaker_id in speakers_to_process:
    print(f"Processing speaker {speaker_id}")
    
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
    test_speaker = speaker_id
    
    
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

    import torch
    import tqdm
    
    # In[6]:
    
    
    import os
    MODEL_PATH = f"finetuned_whisper_output/model/torgo_tiny_finetune_{speaker_id}/pytorch_model.bin"
    
    if os.path.exists(MODEL_PATH):
        print("The file exists.")
    else:
        print("The file does not exist.")
    
    
    # In[7]:
    
    
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
    
    
    # In[10]:
    
    
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
    
    
    # In[11]:
    
    
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
    
    
    # In[12]:
    
    
    # Remove special characters from the text
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\`\�0-9]'
    
    
    def remove_special_characters(batch):
        batch['text'] = re.sub(chars_to_ignore_regex,
                               ' ', batch['text']).lower()
        return batch
    
    torgo_dataset = torgo_dataset.map(remove_special_characters)
    
    
    # In[13]:
    
    
    print(torgo_dataset['train'][2]['text'])
    
    
    # In[14]:
    
    
    import whisper_openAI.whisper as whisper
    from tqdm import tqdm
    from jiwer import wer
    
    # Function to process each audio file
    def process_audio(file_path):
        # Load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(file_path)
        audio = whisper.pad_or_trim(audio)
        
        # Make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        
        # Decode the audio
        options = whisper.DecodingOptions(fp16=False, without_timestamps=True, language='english')
        result, _ = whisper.decode(model, mel, options)
        
        result_text = ''.join(result)
        
        # Print the recognized text
        # print(result_text)
        return result_text
    
    # Initialize the results and ground truth lists
    recognized_texts = []
    ground_truth_texts = []
    
    def normalize_text(text):
        chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\`\�0-9]'
        text = re.sub(chars_to_ignore_regex, ' ', text).lower()
        return text
        
    # Iterate over the dataset
    for i in tqdm(range(len(torgo_dataset['test'])), desc="Processing audio files"):
        # Get the file path and ground truth from the dataset
        file_path = torgo_dataset['test'][i]['audio']
        ground_truth = torgo_dataset['test'][i]['text']
        
        # Process the audio file
        recognized_text = process_audio(file_path)
        recognized_text = normalize_text(recognized_text)
        ground_truth = normalize_text(ground_truth)
        
        # Append the results to the lists
        recognized_texts.append(recognized_text)
        ground_truth_texts.append(ground_truth)
        
        # Print the recognized text
        # print(f"text {i+1}/{len(torgo_dataset['test'])}: {recognized_text}")
        # print(f"Ground truth: {ground_truth}")
        # print()
    
    # Calculate WER for each recognized text against the ground truth
    wer_scores = [wer(gt, rt) for gt, rt in zip(ground_truth_texts, recognized_texts)]
    
    # Print the average WER
    average_wer = sum(wer_scores) / len(wer_scores)
    print(f"Average WER: {average_wer}")

    # Ensure the directory exists
    output_dir = f'runs/full_finetuned'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Optional: Save the recognized texts and ground truths to files
    with open(f'{output_dir}/finetuned_{speaker_id}_recognized_texts.txt', 'w') as f:
        for text in recognized_texts:
            f.write(f"{text}\n")
    
    with open(f'{output_dir}/finetuned_{speaker_id}_ground_truth_texts.txt', 'w') as f:
        for text in ground_truth_texts:
            f.write(f"{text}\n")
    with open(f'{output_dir}/finetuned_{speaker_id}_wer.txt', 'w') as f:
        f.write(f"Average WER: {average_wer}\n")
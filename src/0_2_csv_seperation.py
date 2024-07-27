#!/usr/bin/env python
# coding: utf-8

# In[5]:


import csv
from pathlib import Path

def extract_and_save_fields(csv_file_path, word_output_file_path, sentence_output_file_path, wer_output_file_path):
    # Initialize lists to store the extracted fields
    word_level = []
    sentence_level = []
    wer_gtms = []

    # Read the CSV file
    with open(csv_file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        
        # Iterate over each row in the CSV
        for row in reader:
            inference = row[0]
            ground_truth = row[1]
            wer = row[2] if len(row) > 2 else ""  # Assuming WER is the third column, if present

            # Determine if the fields are word-level or sentence-level
            if len(ground_truth.split()) > 1:
                sentence_level.append(f"{inference},{ground_truth}\n")
            else:
                word_level.append(f"{inference},{ground_truth}\n")
            
            # Prepare the WER field for the output file
            if wer:
                wer_gtms.append(f"{wer}\n")

    # Save the word-level fields to the output file
    with open(word_output_file_path, 'w') as word_output_file:
        word_output_file.writelines(word_level)

    # Save the sentence-level fields to the output file
    with open(sentence_output_file_path, 'w') as sentence_output_file:
        sentence_output_file.writelines(sentence_level)

    # Save the WER field to the output file
    if wer_gtms:
        with open(wer_output_file_path, 'w') as wer_output_file:
            wer_output_file.writelines(wer_gtms)


# List of speakers
speakers = ['F01', 'F03', 'F04', 'M01', 'M02', 'M03', 'M04', 'M05']


# Iterate over each speaker
for speaker in speakers:
    print(f"speaker {speaker}:")
    folder_path = f'/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/Whispering-LLaMA/runs/naive_inference_TORGO_keep_all'
    base_filename = f'finetuned_{speaker}_texts'
    
    # Define the file paths
    csv_file_path = f'{folder_path}/{base_filename}.csv'
    word_output_file_path = f'{folder_path}/{base_filename}_word_inference_gt.txt'
    sentence_output_file_path = f'{folder_path}/{base_filename}_sentence_inference_gt.txt'
    wer_output_file_path = f'{folder_path}/{base_filename}_wer_gtms.txt'
    
    # Extract and save the fields
    extract_and_save_fields(csv_file_path, word_output_file_path, sentence_output_file_path, wer_output_file_path)
    
    print(f"Data extracted and saved to {word_output_file_path}, {sentence_output_file_path}, and {wer_output_file_path}")
    
    
    # In[6]:
    from jiwer import wer
    
    def calculate_wer_from_file(file_path):
        # Read the file and separate inferences and ground truths
        with open(file_path, 'r') as file:
            lines = file.readlines()
    
        inferences = []
        ground_truths = []
        for line in lines:
            if ',' in line:
                inference, ground_truth = line.strip().split(',', 1)
                if ground_truth:  # Check if ground_truth is not empty
                    inferences.append(inference)
                    ground_truths.append(ground_truth)
    
        # Calculate WER using jiwer
        return wer(ground_truths, inferences), len(inferences), len(ground_truths)
        
    
    print(calculate_wer_from_file(word_output_file_path))
    
    print(calculate_wer_from_file(sentence_output_file_path))
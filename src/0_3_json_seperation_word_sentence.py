import json
from pathlib import Path
from jiwer import wer

def extract_and_save_fields(json_file_path, word_output_file_path, sentence_output_file_path, wer_output_file_path, all_output_file_path):
    # Read the JSON file
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
        
    # Initialize lists to store the extracted fields
    word_level = []
    sentence_level = []
    wer_gtms = []
    all_levels = []

    # Iterate over each item in the list
    for item in data:
        inference = item.get("inference", "")
        ground_truth = item.get("ground_truth", "")
        wer_value = item.get("wer", "")

        # Determine if the fields are word-level or sentence-level
        if len(ground_truth.split()) > 1:
            sentence_entry = f"{inference},{ground_truth}\n"
            sentence_level.append(sentence_entry)
            all_levels.append(sentence_entry)
        else:
            word_entry = f"{inference},{ground_truth}\n"
            word_level.append(word_entry)
            all_levels.append(word_entry)
        
        # Prepare the WER field for the output file
        wer_gtms.append(f"{wer_value}\n")

    # Save the word-level fields to the output file
    with open(word_output_file_path, 'w') as word_output_file:
        word_output_file.writelines(word_level)

    # Save the sentence-level fields to the output file
    with open(sentence_output_file_path, 'w') as sentence_output_file:
        sentence_output_file.writelines(sentence_level)

    # Save the WER field to the output file
    with open(wer_output_file_path, 'w') as wer_output_file:
        wer_output_file.writelines(wer_gtms)

    # Save all fields (word and sentence level) to the combined output file
    with open(all_output_file_path, 'w') as all_output_file:
        all_output_file.writelines(all_levels)

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
    return wer(ground_truths, inferences)

# List of speakers
speakers = ['M05']

# Iterate over each speaker
for speaker in speakers:
    folder_path = f'/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/Whispering-LLaMA/runs/Inference/{speaker}'
    base_filename = f'WL_M_0.001_torgo_M05_large-v2iter-018999-loss-0.609.pth'

    # Define the file paths
    json_file_path = f'{folder_path}/{base_filename}.json'
    word_output_file_path = f'{folder_path}/{base_filename}_word_inference.txt'
    sentence_output_file_path = f'{folder_path}/{base_filename}_sentence_inference.txt'
    all_output_file_path = f'{folder_path}/{base_filename}_all_inference.txt'
    wer_output_file_path = f'{folder_path}/{base_filename}_wer_gtms.txt'

    # Extract and save the fields
    extract_and_save_fields(json_file_path, word_output_file_path, sentence_output_file_path, wer_output_file_path,all_output_file_path)

    # Calculate WER for word-level and sentence-level files
    word_wer = calculate_wer_from_file(word_output_file_path)
    sentence_wer = calculate_wer_from_file(sentence_output_file_path)
    all_wer = calculate_wer_from_file(all_output_file_path)

    print(f"Data extracted and saved for {speaker} to {word_output_file_path}, {sentence_output_file_path}, and {wer_output_file_path}")
    print(f"WER for word-level data for {speaker}: {word_wer}")
    print(f"WER for sentence-level data for {speaker}: {sentence_wer}")
    print(f"WER for all-level data for {speaker}: {all_wer}")
# %%
# This notebook uses https://github.com/openai/whisper with edits to the whisper_openAI/decoding.py to generate multiple hypothesis
import datasets
from datasets import load_dataset
import tqdm


# %%
# To print nuber of datapoints per category in the gigaspeech dataset form Hugging Face
# Load the dataset using the correct configuration
dataset = load_dataset("edinburghcstr/ami", "ihm", 
    cache_dir='/work/van-speech-nlp/temp',
    use_auth_token='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'
    )

import sys
from pathlib import Path
# Set the absolute path to the parent directory of whisper_openAI
parent_dir = Path('/work/van-speech-nlp/jindaznb/jslpnb/mllm_expriments/Whispering-LLaMA')  # Adjust this to your actual path
sys.path.append(str(parent_dir))

# Verify the addition
print(sys.path)

# %%
import numpy
# Renamed the Whisepr repo (https://github.com/openai/whisper) with the changed decoding.py file as whisper_openAI
import whisper_openAI.whisper as whisper
import torch
import tqdm
model ,_ = whisper.load_model("large-v2") # you can change the whisper model here to largev2 or large to swap the  model.

# %%
import json
import os
import tqdm
import numpy as np


def save_checkpoint(to_json, index, split_name):
    checkpoint = {'to_json': to_json, 'index': index}
    with open(f'Inference/gs_inferences/ami/{split_name}_checkpoint.pkl', 'wb') as f:
        pickle.dump(checkpoint, f)

def load_checkpoint(split_name):
    checkpoint_path = f'Inference/gs_inferences/ami/{split_name}_checkpoint.pkl'
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        return checkpoint['to_json'], checkpoint['index']
    return [], 0

def generate_inference_json(data_split, split_name):
    to_json, start_index = load_checkpoint(split_name)
    
    for idx in tqdm.tqdm(range(start_index, len(data_split))):
        i = data_split[idx]
        audio = np.array(i['audio']['array']).astype(np.single)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device) 
        ground_truth = i['text'].replace(' <COMMA>', ',').replace(' <PERIOD>', '.').replace(' <QUESTIONMARK>', '?').replace(' <EXCLAMATIONPOINT>', '!').lower()
        source = i.get('source', '')  # Default to empty string if 'source' is not present
        cat = i.get('category', '')   # Default to empty string if 'category' is not present
        time = i['end_time'] - i['begin_time']
        path_to_file = i['audio']['path']
        random_temperature = np.random.randint(70, 81) / 100
        options = whisper.DecodingOptions(fp16=True, without_timestamps=True, temperature=random_temperature, best_of=50)
        result, _ = whisper.decode(model, mel, options)
        result = list(result)

        # redo if results is too less
        if len(result) <= 10:
            if random_temperature < 0.75:
                random_temperature += 0.2
            else:
                random_temperature += 0.1
            options = whisper.DecodingOptions(fp16=True, without_timestamps=True, temperature=random_temperature, best_of=50)
            result, _ = whisper.decode(model, mel, options)
            result = list(result)

        to_json.append({i['audio_id']: {'temp': random_temperature, 'path': path_to_file, 'ground_truth': ground_truth, 'inference': result, 'source': source, 'category': cat, 'time': time, 'path': path_to_file}})
        
        # Save checkpoint every 100 iterations
        if idx % 100 == 0 and idx > 0:
            save_checkpoint(to_json, idx + 1, split_name)
    
    # Save final result
    os.makedirs(f"Inference/gs_inferences/ami", exist_ok=True)

    save_path = f'Inference/gs_inferences/ami/{split_name}.json'
    with open(save_path, "w") as file:
        json.dump(to_json, file, indent=4)
    
    # Remove checkpoint file after completion
    checkpoint_path = f'Inference/gs_inferences/ami/{split_name}_checkpoint.pkl'
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


os.makedirs(f"Inference/gs_inferences/ami", exist_ok=True)

# Generate inferences for each data split
generate_inference_json(dataset['train'], 'train')
generate_inference_json(dataset['validation'], 'validation')
generate_inference_json(dataset['test'], 'test')

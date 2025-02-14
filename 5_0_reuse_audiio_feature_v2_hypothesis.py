#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
from pathlib import Path
import torch
import requests
import json
import os 

model_name="large-v2"



# In[2]:


from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import whisper_openAI.whisper as whisper
import torch
from whisper_openAI.whisper.tokenizer import Tokenizer, get_tokenizer
import torch
import torch.nn.functional as F
from torch import Tensor
import json

# We get the acoustic embeddings from Whisper Large V2
model,processor = whisper.load_model("large-v2")
# model,processor = whisper.load_model("medium")


# In[3]:


model.eval()


speakers_to_process = [
    # F01 training set is wrong
    'F01',  

    # finished
    # 'F03', 
    # 'F04', 
    # 'M01', 
    # 'M02', 
    # 'M03', 
    # 'M04', 
    # 'M05'
]



for speaker_id in speakers_to_process:
    test_speaker=speaker_id
    print(f"Processing speaker {speaker_id}")

    
    # The below is the json file you can generate using the "To generatn-best hyporhesis.ipynb" notebook; Need to further tokenize the hypothesis
    
    with open(f'Inference/gs_inferences/torgo_train_{speaker_id}_{model_name}.json', "r") as file:  # Change the file path and name here
        train_data = json.load(file)
    
    with open(f'Inference/gs_inferences/torgo_val_{speaker_id}_{model_name}.json', "r") as valid_file:
        val_data = json.load(valid_file)
    
    # Load the test set
    with open(f'Inference/gs_inferences/torgo_test_{speaker_id}_{model_name}.json', "r") as test_file:
        test_data = json.load(test_file)
    
    


    
    
    from lit_llama.tokenizer import Tokenizer
    from tqdm import tqdm
    
    
    tokenizer_path: Path = Path("weights/tokenizer.model")
    tokenizer = Tokenizer(tokenizer_path)
    print(f"train has {len(train_data):,} samples")
    
    
    # In[7]:
    
    
    import torch
    old_data_train = torch.load(f'Inference/gs_inferences/baseline_data_tiny_hypo_v2/torgo_{test_speaker}_train.pt',map_location=torch.device('cpu'))
    old_data_val = torch.load(f'Inference/gs_inferences/baseline_data_tiny_hypo_v2/torgo_{test_speaker}_val.pt',map_location=torch.device('cpu'))
    old_data_test = torch.load(f'Inference/gs_inferences/baseline_data_tiny_hypo_v2/torgo_{test_speaker}_test.pt',map_location=torch.device('cpu'))
    
    
    # In[8]:
    
    
    # Print the lengths of the loaded data
    print(f'Length of old_data_train: {len(old_data_train)}')
    print(f'Length of old_data_val: {len(old_data_val)}')
    print(f'Length of old_data_test: {len(old_data_test)}')
    
    
    # In[9]:
    
    
    def tokenize(tokenizer: Tokenizer, string: str, max_length: int, eos=True) -> torch.Tensor:
        return tokenizer.encode(string, bos=True, eos=eos, max_length=max_length)
        
    def process_train_data(train_data, old_data):
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
            
            # path = ip['path']
            # audio = whisper.load_audio(path)  
            # audio = whisper.pad_or_trim(audio)            
            # mel = whisper.log_mel_spectrogram(audio).to(model.device)  # Adjust as needed for your model
            # mel = mel.unsqueeze(0)
            
            # with torch.no_grad():
            #     audio_features = model.encoder(mel)
            audio_features = old_data[i]["audio_features"]
            
            result.append({**ip, 'index': name, "input_ids": encoded_full_prompt_and_response, "input_ids_no_response": encoded_full_prompt, "labels": labels, 'labels_with_masked_input': labels_with_masked_input, 'audio_features': audio_features.bfloat16()})
        print(len(result))
        return result
    
    
    # In[10]:
    
    
    split = "train"
    result = process_train_data(train_data, old_data_train)
    torch.save(result,f'Inference/gs_inferences/torgo_{speaker_id}_{model_name}_{split}.pt')
    print(f"Processed {split} data and saved checkpoint for {speaker_id}")
    
    split = "val"
    result = process_train_data(val_data, old_data_val)
    torch.save(result,f'Inference/gs_inferences/torgo_{speaker_id}_{model_name}_{split}.pt')
    print(f"Processed {split} data and saved checkpoint for {speaker_id}")
    
    split = "test"
    result = process_train_data(test_data, old_data_test)
    torch.save(result,f'Inference/gs_inferences/torgo_{speaker_id}_{model_name}_{split}.pt')
    print(f"Processed {split} data and saved checkpoint for {speaker_id}")
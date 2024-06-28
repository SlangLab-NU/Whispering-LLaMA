import time  # Import the time module at the beginning

# Record the start time
start_time = time.time()

# https://github.com/SlangLab-NU/torgo_inference_on_cluster/blob/main/train.py
# https://medium.com/@shridharpawar77/a-comprehensive-guide-for-custom-data-fine-tuning-with-the-whisper-model-60e4cbce736d
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments,TrainingArguments
from transformers import WhisperForConditionalGeneration
import evaluate
from typing import Any, Dict, List, Union
from transformers import WhisperProcessor
from transformers import WhisperTokenizer
from transformers import WhisperFeatureExtractor
import sys
import os
import argparse
import re
import json
import torch
import logging
import pandas as pd
import numpy as np

from datasets import load_dataset, DatasetDict, Audio
from dataclasses import dataclass
from typing import Dict, List, Union
from evaluate import load
from tqdm import tqdm
from datetime import datetime

# get_ipython().system('huggingface-cli login --token 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')

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
                    default='', help='name of model to be trained on')

args = parser.parse_args()

# Check if the speaker ID is valid
if not re.match(r'^[MF]C?[0-9]{2}$', args.speaker_id):
    print("Please provide a valid speaker ID.")
    sys.exit(1)
test_speaker = args.speaker_id

# Accessing optional arguments
speaker_id = args.speaker_id
debug = args.debug
learning_rate = args.learning_rate
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
seed = args.seed
gradient_accumulation_steps = args.gradient_accumulation_steps
optimizer = args.optimizer
lr_scheduler_type = args.lr_scheduler_type
num_epochs = args.num_epochs
keep_all_data = args.keep_all_data
debug_mode = args.debug
repo_suffix = args.repo_suffix
model_name = args.model_name
if args.repo_suffix and not re.match(r'^[_-]', args.repo_suffix):
    repo_suffix = '_' + args.repo_suffix

print("Speaker ID: " + speaker_id)
print("Learning rate: {learning_rate}")
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

if not re.match(r'^[MF]C?[0-9]{2}$', speaker_id):
    print("Please provide a valid speaker ID.")
    sys.exit(1)
test_speaker = speaker_id

if repo_suffix and not re.match(r'^[_-]', args.repo_suffix):
    repo_suffix = '_' + repo_suffix


# Define the path to the CSV file
torgo_csv_path = "./torgo.csv"

# Check if the path exists and is a file
if os.path.exists(torgo_csv_path) and os.path.isfile(torgo_csv_path):
    print("The CSV file exists.")
else:
    print("The CSV file does not exist.")

torgo_dataset_path = '/work/van-speech-nlp/data/torgo'
torgo_dataset_dir_path = torgo_dataset_path + \
    '/' if torgo_dataset_path[-1] != '/' else torgo_dataset_path
output_path = '../finetuned_whisper_output'
print(f'torgo_dataset_path: {torgo_dataset_path}')
print(f'torgo_dataset_dir_path: {torgo_dataset_dir_path}')

repo_name = f'torgo_{model_name}_finetune_{test_speaker}{repo_suffix}_frozen_encoder'
repo_path = f'jindaxz/{repo_name}'

# Path to save model / checkpoints{repo_name}'
model_local_path = output_path + '/model/' + repo_name

pretrained_model_name = f"openai/{model_name}"


# In[5]:


if not os.path.exists(output_path + '/logs'):
    os.makedirs(output_path + '/logs')

log_dir = f'{output_path}/logs/{repo_name}'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file_name = test_speaker + '_train' + '_' + \
    datetime.now().strftime("%Y%m%d_%H%M%S") + '.log'
log_file_path = log_dir + '/' + log_file_name

logging.basicConfig(
    filename=log_file_path,
    filemode='a',
    format='%(asctime)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.INFO
)
# Log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)

logging.info("Test Speaker: " + test_speaker)
logging.info("Log File Path: " + log_file_path + '\n')
if keep_all_data:
    logging.info("Keep all data in training/validation/test sets\n")


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


# In[ ]:


# Remove special characters from the text
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\`\�0-9]'


def remove_special_characters(batch):
    batch['text'] = re.sub(chars_to_ignore_regex,
                           ' ', batch['text']).lower()
    return batch


torgo_dataset = torgo_dataset.map(remove_special_characters)


# In[9]:


# convert the sample rate of every audio files using cast_column function
torgo_dataset = torgo_dataset.cast_column("audio", Audio(sampling_rate=16000))


# In[10]:


# Define the minimum and maximum input length in seconds
min_input_length_in_sec = 1.0
max_input_length_in_sec = 10.0
sampling_rate = 16000

# Define the filtering functions based on input length


def filter_min_length(example):
    return example["audio"]["array"].shape[0] > min_input_length_in_sec * sampling_rate


def filter_max_length(example):
    return example["audio"]["array"].shape[0] < max_input_length_in_sec * sampling_rate


# Apply the filters
torgo_dataset = torgo_dataset.filter(filter_max_length)
torgo_dataset = torgo_dataset.filter(filter_min_length)


# In[11]:


logging.info(
    f"After filter, the number of data in each dataset is:")
logging.info(
    f'Train:       {len(torgo_dataset["train"])}/{original_data_count["train"]} ({len(torgo_dataset["train"]) * 100 // original_data_count["train"]}%)')
logging.info(
    f'Validation:  {len(torgo_dataset["validation"])}/{original_data_count["validation"]} ({len(torgo_dataset["validation"]) * 100 // original_data_count["validation"]}%)')
logging.info(
    f'Test:        {len(torgo_dataset["test"])}/{original_data_count["test"]} ({len(torgo_dataset["test"]) * 100 // original_data_count["test"]}%)\n')


# In[12]:


# import feature extractor

feature_extractor = WhisperFeatureExtractor.from_pretrained(
    f"openai/{model_name}")

# Load WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained(
    f"openai/{model_name}", language="English", task="transcribe")
# Combine To Create A WhisperProcessor

processor = WhisperProcessor.from_pretrained(
    f"openai/{model_name}", language="English", task="transcribe")


# In[13]:


def prepare_dataset(examples):
    # compute log-Mel input features from input audio array
    audio = examples["audio"]
    examples["input_features"] = processor(
        audio["array"], sampling_rate=16000).input_features[0]
    examples["input_length"] = len(examples["input_features"])
    # print(examples["input_length"])
    del examples["audio"]
    sentences = examples["text"]

    # encode target text to label ids
    examples["labels"] = tokenizer(sentences).input_ids
    del examples["text"]
    return examples


# In[14]:


torgo_dataset = torgo_dataset.map(prepare_dataset, num_proc=4)


# In[15]:


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]}
                          for feature in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt")
        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]}
                          for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt")
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)
        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


# lets initiate the data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


# In[16]:


torgo_dataset = torgo_dataset.remove_columns(["input_length"])


# In[17]:


metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# In[18]:


# Load a Pre-Trained Checkpoint
model = WhisperForConditionalGeneration.from_pretrained(f"openai/{model_name}")


# Freeze the feature encoder
for name, param in model.named_parameters():
    if 'encoder' in name:  # Assuming 'encoder' is in the name of the feature encoder layers
        param.requires_grad = False

# In[19]:


model.config.forced_decoder_ids = None
model.config.suppress_tokens = []


training_args = Seq2SeqTrainingArguments(
    output_dir=f"{model_local_path}",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=0.0001,
    warmup_steps=1000,
    max_steps=-1,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=1,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    # logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    num_train_epochs=num_epochs,
    push_to_hub=False,
)


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=torgo_dataset['train'],
    eval_dataset=torgo_dataset['validation'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)


# In[21]:


trainer.create_model_card(
    language="en",
    tags=["audio", "speech", "whisper"],
    model_name=repo_name,
    finetuned_from=f'{pretrained_model_name}',
    tasks=["automatic-speech-recognition"],
    dataset="torgo",
)


# In[22]:


checkpoint_files = [f for f in os.listdir(output_path + '/model/' + repo_name) if f.startswith(
    'checkpoint-') and os.path.isdir(output_path + '/model/' + repo_name + '/' + f)]
if len(checkpoint_files) == 0:
    logging.info(
        "No checkpoint found in the repository. Training from scratch.")
    trainer.train()
else:
    logging.info(
        f"Checkpoint found in the repository. Checkpoint files found: {checkpoint_files}")
    resume_from_checkpoint = f"{model_local_path}/{checkpoint_files[-1]}"
    logging.info(f"Resuming from checkpoint: {resume_from_checkpoint}\n")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


# In[ ]:


logging.info("Pushing model to Hugging Face...")
trainer.push_to_hub()


# In[ ]:


end_time = time.time()
elapsed_time = end_time - start_time
elapsed_minutes = elapsed_time / 60

print(f"The script took {elapsed_minutes:.2f} minutes to run.")

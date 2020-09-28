"""Training function."""

import os
import sys
import json
import torch

from transformers import TrainingArguments
from arguments import DataTrainingArguments, ModelArguments
from model import SentNewsTrainer


if __name__ == '__main__':

    TASK_NAME = "st_news"
    NUM_LABELS = 3
    MAX_SEQUENCE_LENGTH = 32
    OUTPUT_MODE = "classification"
    DATA_DIR = "data/processed"
    MODEL_NAME_OR_PATH = "./multilingual_sentiment_vocab20k"
    OUTPUT_DIR = "/tmp/{}/".format(TASK_NAME)
    OVERWRITE_CACHE = True
    OVERWRITE_OUTPUT_DIR = True

    data_args = DataTrainingArguments(
        task_name=TASK_NAME, 
        data_dir=DATA_DIR, 
        max_seq_length=MAX_SEQUENCE_LENGTH, 
        num_labels=NUM_LABELS,
        output_mode=OUTPUT_MODE,
        overwrite_cache=OVERWRITE_CACHE
    )

    model_args = ModelArguments(
        model_name_or_path=MODEL_NAME_OR_PATH, 
        config_name=None, 
        tokenizer_name=None, 
        cache_dir=None
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR, 
        overwrite_output_dir=OVERWRITE_OUTPUT_DIR, 
        do_train=True, 
        do_eval=True, 
        do_predict=False, 
        evaluate_during_training=True, 
        per_device_train_batch_size=32, 
        per_device_eval_batch_size=128, 
        per_gpu_train_batch_size=None, 
        per_gpu_eval_batch_size=None, 
        gradient_accumulation_steps=1, 
        learning_rate=2e-05, 
        weight_decay=0.0, 
        adam_epsilon=1e-08, 
        max_grad_norm=1.0, 
        num_train_epochs=2, 
        max_steps=-1, 
        warmup_steps=0, 
        logging_dir="logging", 
        logging_first_step=False, 
        logging_steps=500, 
        save_steps=100, 
        eval_steps=50, 
        save_total_limit=None, 
        seed=42, 
        fp16=False, 
        fp16_opt_level="O1", 
        local_rank=-1, 
        debug=False, 
        dataloader_drop_last=False, 
        past_index=-1
    )

    trainer = SentNewsTrainer(data_args, model_args, training_args)
    trainer.train()

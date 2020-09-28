"""Training function."""

import os
import sys
import json
import torch

from transformers import TrainingArguments, AutoTokenizer
from arguments import DataTrainingArguments, ModelArguments
from model import SentNewsTrainer
from processors import SentNewsDataset


if __name__ == '__main__':

    TASK_NAME = "st_news"
    NUM_LABELS = 3
    MAX_SEQUENCE_LENGTH = 32
    OUTPUT_MODE = "classification"
    DATA_DIR = "data/processed"
    CKPT_DIR = "/tmp/{}/".format(TASK_NAME)
    OUTPUT_DIR = "data/prediction"

    data_args = DataTrainingArguments(
        task_name=TASK_NAME, 
        data_dir=DATA_DIR, 
        max_seq_length=MAX_SEQUENCE_LENGTH, 
        num_labels=NUM_LABELS,
        output_mode=OUTPUT_MODE,
        overwrite_cache=False
    )

    model_args = ModelArguments(
        model_name_or_path=CKPT_DIR, 
        config_name=None, 
        tokenizer_name=None, 
        cache_dir=None
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        do_train=False, 
        do_eval=True, 
        do_predict=False, 
        per_device_eval_batch_size=128,
        logging_dir="logging"
    )

    # 1. Evaluate dataset when setting --do_eval as True
    trainer = SentNewsTrainer(data_args, model_args, training_args)
    trainer.evaluate(save_result=True)

    # 2. Evaluate dataset and pass it to evaluate function
    # trainer = SentNewsTrainer(data_args, model_args, training_args)
    # eval_dataset = SentNewsDataset(data_args, tokenizer=trainer.tokenizer, mode="dev")
    # trainer.evaluate(eval_dataset=eval_dataset, save_result=True)

    # 3. Predict dataset when setting --do_predict as True
    # trainer = SentNewsTrainer(data_args, model_args, training_args)
    # trainer.predict(save_result=True)
    
    # 4. Predict dataset and pass it to predict function
    # trainer = SentNewsTrainer(data_args, model_args, training_args)
    # test_dataset = SentNewsDataset(data_args, tokenizer=trainer.tokenizer, mode="test")
    # trainer.predict(test_dataset=test_dataset, save_result=True)

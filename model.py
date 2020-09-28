"""Trainer class, including train, evaluate and test functions."""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from typing import Callable, Dict, Optional
from collections import OrderedDict
from copy import deepcopy

import torch
from transformers import EvalPrediction
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers import Trainer, TrainingArguments, set_seed

from processors import SentNewsDataset

logger = logging.getLogger(__name__)


def build_compute_metrics_fn(output_mode: str) -> Callable[[EvalPrediction], Dict]:
    def compute_metrics_fn(p: EvalPrediction):
        if output_mode == "classification":
            preds = np.argmax(p.predictions, axis=1)
            return {"acc": (preds == p.label_ids).mean()}
        elif output_mode == "regression":
            preds = np.squeeze(p.predictions)
            raise NotImplementedError('Regression metrics not implemented')
    return compute_metrics_fn


class SentNewsTrainer(object):

    def __init__(self, data_args, model_args, training_args):
        if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
        ):
            raise ValueError(
                f"Directory {training_args.output_dir} is not empty. Use `overwrite_output_dir` to overwrite."
            )
        self.data_args = data_args
        self.model_args = model_args
        self.training_args = training_args

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if self.training_args.local_rank in [-1, 0] else logging.WARN,
        )
        logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            self.training_args.local_rank,
            self.training_args.device,
            self.training_args.n_gpu,
            bool(self.training_args.local_rank != -1),
            self.training_args.fp16,
        )
        logger.info("Training/evaluation parameters %s", self.training_args)

        # Set seed
        set_seed(self.training_args.seed)

        # Load tokenizer and pretrained model
        self.config = AutoConfig.from_pretrained(
            self.model_args.config_name if self.model_args.config_name else self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name if self.model_args.tokenizer_name else self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
        )
    
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_args.model_name_or_path,
            from_tf=False,
            config=self.config,
            cache_dir=self.model_args.cache_dir,
        )

        # Set ouput size equal to number of labels
        if self.data_args.num_labels != self.model.classifier.out_features:
            in_features = self.model.classifier.in_features
            self.model.classifier = torch.nn.Linear(in_features=in_features, out_features=data_args.num_labels)
            self.config.num_labels = self.data_args.num_labels
            self.model.config = self.config
            self.model.num_labels = self.data_args.num_labels
    
        # Get datasets
        self.train_dataset = (
            SentNewsDataset(self.data_args, tokenizer=self.tokenizer, cache_dir=self.model_args.cache_dir) 
            if self.training_args.do_train else None
        )
        self.eval_dataset = (
            SentNewsDataset(self.data_args, tokenizer=self.tokenizer, mode="dev", cache_dir=self.model_args.cache_dir)
            if self.training_args.do_eval else None
        )
        self.test_dataset = (
            SentNewsDataset(self.data_args, tokenizer=self.tokenizer, mode="test", cache_dir=self.model_args.cache_dir)
            if self.training_args.do_predict else None
        )
        
        # Initialize our Trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=build_compute_metrics_fn(self.data_args.output_mode),
        )


    def train(self):
        if self.training_args.do_train:
            self.trainer.train(
                model_path=self.model_args.model_name_or_path 
                if os.path.isdir(self.model_args.model_name_or_path) else None
            )
            self.trainer.save_model()
            if self.trainer.is_world_master():
                self.tokenizer.save_pretrained(self.training_args.output_dir)

        if self.training_args.do_eval:
            eval_result = self.evaluate(return_result=True)

        if self.training_args.do_predict:
            self.predict(save_result=True)


    def evaluate(self, eval_dataset=None, return_result=False, save_result=False):
        if not any([self.eval_dataset, eval_dataset]):
            raise ValueError("Must set --do_eval as True in TrainingArguments"
            " or provide argument `eval_dataset`"
        )
        eval_dataset = eval_dataset if eval_dataset else self.eval_dataset
        eval_result = None
        if self.training_args.do_train:
            self.trainer.compute_metrics = build_compute_metrics_fn(self.data_args.output_mode)
            eval_result = self.trainer.evaluate(eval_dataset=eval_dataset)

            if self.trainer.is_world_master():
                output_eval_summary_file = os.path.join(
                    self.training_args.output_dir, f"{self.data_args.task_name}_eval_results.txt"
                )
                with open(output_eval_summary_file, "w") as writer:
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
        
        if self.trainer.is_world_master() and save_result:
            output_eval_detail_file = os.path.join(
                self.training_args.output_dir, f"{self.data_args.task_name}_eval_results.xlsx"
            )
            predictions = self.trainer.predict(test_dataset=eval_dataset).predictions

            examples = np.array([[e.text_a, e.label] for e in eval_dataset.examples])
            examples_ids = np.argmax(predictions, axis=1)[:, np.newaxis]
            softmax = torch.nn.Softmax(dim=1)
            examples_prob = softmax(
                torch.Tensor(predictions)
            ).max(axis=1).values.numpy()[:, np.newaxis]

            examples = np.concatenate((examples, examples_ids, examples_prob), axis=1)
            df = pd.DataFrame(
                examples, 
                columns=['sentence', 'label', 'pred_label', 'probability']
            )

            df['pred_label'] = df['pred_label'].map(lambda x: eval_dataset.get_labels()[int(x)])
            df.to_excel(output_eval_detail_file, index=False)

        if return_result:
            return eval_result


    def predict(self, test_dataset=None, return_result=False, save_result=False):
        if not any([self.test_dataset, test_dataset]):
            raise ValueError("Must set --do_predict as True in TrainingArguments"
            " or provide argument `test_dataset`"
        )
        test_dataset = test_dataset if test_dataset else self.test_dataset
        predictions = self.trainer.predict(test_dataset=test_dataset).predictions

        if self.trainer.is_world_master() and save_result:
            output_test_detail_file = os.path.join(
                self.training_args.output_dir, f"{self.data_args.task_name}_test_results.xlsx"
            )

            examples = np.array([[e.text_a] for e in test_dataset.examples])
            examples_ids = np.argmax(predictions, axis=1)[:, np.newaxis]
            softmax = torch.nn.Softmax(dim=1)
            examples_prob = softmax(
                torch.Tensor(predictions)
            ).max(axis=1).values.numpy()[:, np.newaxis]

            examples = np.concatenate((examples, examples_ids, examples_prob), axis=1)
            df = pd.DataFrame(
                examples, 
                columns=['sentence', 'pred_label', 'probability']
            )

            df['pred_label'] = df['pred_label'].map(lambda x: test_dataset.get_labels()[int(x)])
            df.to_excel(output_test_detail_file, index=False)

        if return_result:
            return predictions


from pprint import pprint

from typing import List, Dict
from unicodedata import normalize
import json
import math
import numpy as np
import pandas as pd
import evaluate
from datasets import (Dataset, load_dataset, Features, Value, ClassLabel)
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          TrainingArguments,
                          Trainer)
from transformers.trainer_utils import BestRun


class Preprocessor():

    class_names: List[str] = [
        'Opportunity',
        'Risk'
    ]
    esg_features: Features = Features({
        'sentence': Value('string'),
        'label': ClassLabel(names=class_names)
    })

    def __init__(
            self,
            input_path: str = None, 
            train_path: str = None,
            test_path: str = None,
            augment_path: str = None, 
            translated_path: str = None
            ):
        print('initialize preprocessor')
        self.augment_path = augment_path
        self.translated_path = translated_path
        if input_path is not None:
            self.input_path = input_path
            self.read_input()
            self.train_test_split(test_size=0.2, seed=47)
        elif train_path is not None and test_path is not None:
            self.train_path = train_path
            self.test_path = test_path
            self.read_train_test()
            
    
    def read_input(self) -> None:
        df = pd.read_json(self.input_path)
        df = df.rename(
            columns={
                'news_content': 'sentence',
                'impact_type': 'label'
            }
        )[['sentence', 'label']]
        df['sentence'] = df['sentence'].apply(
            Preprocessor.clean_text
            )
        ds = Dataset.from_pandas(
            df,
            features=Preprocessor.esg_features
            )
        self.input_dataset = ds

    def train_test_split(self, test_size: float, seed: int) -> None:
        self.input_dataset = self.input_dataset.train_test_split(
            test_size=test_size,
            stratify_by_column='label',
            seed=seed
        )

    def read_train_test(self) -> None:
        train_paths = [self.train_path]
        if self.augment_path is not None:
            train_paths.append(self.augment_path)
        if self.translated_path is not None:
            train_paths.append(self.translated_path)
        self.input_dataset = load_dataset(
            'json',
            data_files={
                'train': train_paths,
                'test': [self.test_path]
            },
            features=Preprocessor.esg_features
        )

    @staticmethod
    def clean_text(text: str) -> str:
        return normalize('NFKD', text)


class Model():

    def __init__(
            self,
            model_name: str,
            prep_data: Preprocessor,
            output_alias: str = None,
            eval_metric: str = 'f1',
            learning_rate: float = 2.5e-5,
            batch_size: int = 16,
            weight_decay: float = 0.01,
            num_epochs: int = 2,
            seed: int = 28
            ):
        print('initialize model')
        self.model_name = model_name
        self.metric = evaluate.load(eval_metric)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = self.model_init()
        self.output_alias = f'{self.model_name}-ft-esg1' if output_alias is None else output_alias
        self.args = TrainingArguments(
            self.output_alias,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            seed=seed,
            load_best_model_at_end=True,
            metric_for_best_model=eval_metric,
            push_to_hub=False
        )
        self.prep_data = prep_data
        self.enc_data = self.encode()
        self.train_dataset = self.enc_data['train']
        self.eval_dataset = self.enc_data['test']
        self.trainer = Trainer(
            self.model,
            self.args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_eval
        )
        self.tuner = Trainer(
            model_init=self.model_init,
            args=self.args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_eval
        )
        self.best_run = None

    def model_init(self):
        return AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            ignore_mismatched_sizes=True
        )

    def tokenize(self, examples):
        return self.tokenizer(examples['sentence'], truncation=True)
    
    def encode(self) -> Dataset:
        return self.prep_data.input_dataset.map(self.tokenize, batched=True)
    
    def compute_eval(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.metric.compute(
            predictions=predictions, 
            references=labels
            )

    def train(self):
        return self.trainer.train()
    
    def evaluate(self):
        return self.trainer.evaluate()
    
    @staticmethod
    def hp_space_optuna(batch_size: List[int] = None):
        if batch_size is None:
            batch_size = [4, 8, 16]

        print(batch_size)
        
        def fn_hp_space_optuna(trial) -> Dict[str, float]:
            return {
                'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-4, 0.1, log=True),
                'num_train_epochs': trial.suggest_int('num_train_epochs', 1, 5),
                'per_device_train_batch_size': trial.suggest_categorical('per_device_train_batch_size', batch_size),
            }
        
        return fn_hp_space_optuna
    
    def tune(self, n_trials: int, max_batch_size: int) -> BestRun:
        batch_size = [2 ** n for n in range(2, max(2, int(math.log2(max_batch_size)))+1)]
        self.best_run = self.tuner.hyperparameter_search(
            backend='optuna',
            hp_space=Model.hp_space_optuna(batch_size),
            n_trials=n_trials, 
            direction='maximize'
        )
        return self.best_run
    

class Trial():
    
    def __init__(
            self,
            model_name: str,
            max_batch_size: str = None,
            input_path: str = None,
            train_path: str = None,
            test_path: str = None,
            augment_path: str = None,
            translated_path: str = None
            ):
        self.model_name = model_name
        if input_path is not None:
            self.preprocessor = Preprocessor(
                input_path=input_path
            )
        else:
            self.preprocessor = Preprocessor(
                train_path=train_path,
                test_path=test_path,
                augment_path=augment_path,
                translated_path=translated_path
            )
        self.model = Model(
            model_name,
            prep_data=self.preprocessor
        )
        self.max_batch_size = max_batch_size

    def run_once(self):
        self.model.train()
        return self.model.evaluate()

    def run_optimize(self, n_trials=5):
        self.model.tune(n_trials=n_trials, max_batch_size=self.max_batch_size)
        return self.model.best_run
    

if __name__=='__main__':

    # French Inference

    model_ids = [
        'distilbert-base-uncased'
        'roberta-large',
        'microsoft/deberta-v3-base'
    ]
    for model_id in model_ids:
        cv_runs = {}
        for i in range(5):
            print(f"run-{i+1}")
            trial_arg = {
                'model_name': model_id,
                'train_path': f'esg-finnlp/data/splits/fr/{i+1}_train.json',
                'test_path': f'esg-finnlp/data/splits/fr/{i+1}_test.json',
                # 'augment_path': f'data/splits/en/{i+1}/augment.json'
            }
            trial = Trial(**trial_arg)
            results = trial.run_once()
            cv_runs[str(i)] = results
        with open(f'{model_id}_fr.json', 'w') as fp:
            json.dump(cv_runs, fp, indent=2)

    # English Inference

    for model_id in model_ids:
        cv_runs = {}
        for i in range(5):
            print(f"run-{i+1}")
            trial_arg = {
                'model_name': model_id,
                'train_path': f'data/splits/en/{i+1}/train.json',
                'test_path': f'data/splits/en/{i+1}/test.json',
                # 'translated_path': f'data/splits/en/{i+1}/fr.json'
            }
            trial = Trial(**trial_arg)
            results = trial.run_once()
            cv_runs[str(i)] = results
        with open(f'{model_id}_en_tr.json', 'w') as fp:
            json.dump(cv_runs, fp, indent=2)

    # Hyperparameter Tuning

    INPUT_EN = 'esg-finnlp/data/raw/ML-ESG-2_English_Train.json'
    trial_args = [
        {
            'model_name': 'distilbert-base-uncased',
            'input_path': INPUT_EN,
            'max_batch_size': 64
        },
        {
            'model_name': 'roberta-base',
            'input_path': INPUT_EN,
            'max_batch_size': 8
        },
        {
            'model_name': 'microsoft/deberta-v3-base',
            'input_path': INPUT_EN,
            'max_batch_size': 8
        },
        {
            'model_name': 'roberta-large',
            'input_path': INPUT_EN,
            'max_batch_size': 8
        },
        {
            'model_name': 'microsoft/deberta-v3-large',
            'input_path': INPUT_EN,
            'max_batch_size': 8
        }
    ]
    results = []
    for trial_arg in trial_args:
        print(trial_arg['model_name'])
        trial = Trial(**trial_arg)
        best_run = trial.run_optimize()
        results.append({
            'trial_arg': trial_arg,
            'objective': best_run.objective,
            'hyperparameters': best_run.hyperparameters
        })
    with open('trial-berta.log', 'w') as log_file:
        pprint.pprint(results, log_file)

from typing import List
from unicodedata import normalize
import numpy as np
import pandas as pd
import evaluate
from datasets import (Dataset, Features, Value, ClassLabel)
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
            input_path: str, 
            augment_path: str = None, 
            tranlated_paths: str = None
            ):
        print('initialize preprocessor')
        self.input_path = input_path
        self.augment_path = augment_path
        self.translated_paths = tranlated_paths
        self.read_input()
        self.train_test_split(test_size=0.2, seed=47)

    
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
            num_epochs: int = 5,
            seed: int = 1337
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
            num_labels=2
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
    
    def tune(self, n_trials: int) -> BestRun:
        self.best_run = self.tuner.hyperparameter_search(n_trials=n_trials, direction='maximize')
        return self.best_run
    

class Trial():
    
    def __init__(
            self,
            model_name: str,
            input_path: str      
            ):
        self.model_name = model_name
        self.input_path = input_path
        self.preprocessor = Preprocessor(
            input_path=self.input_path
        )
        self.model = Model(
            model_name,
            prep_data=self.preprocessor
        )

    def run_once(self):
        self.model.train()
        self.model.evaluate()

    def run_optimize(self, n_trials=10):
        self.model.tune(n_trials=n_trials)
        return self.model.best_run
    

from pprint import pprint
if __name__=='__main__':
    INPUT_EN = 'esg-finnlp/data/raw/ML-ESG-2_English_Train.json'
    trial_args = [
        {
            'model_name': 'distilbert-base-uncased',
            'input_path': INPUT_EN
        },
        {
            'model_name': 'roberta-base',
            'input_path': INPUT_EN
        }
    ]
    results = []
    for trial_arg in trial_args:
        trial = Trial(**trial_arg)
        best_run = trial.run_optimize()
        results.append({
            'trial_arg': trial_arg,
            'objective': best_run.objective,
            'hyperparameters': best_run.hyperparameters
        })

from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding, EarlyStoppingCallback
from pathlib import Path
import numpy as np
import evaluate
import torch
from pynvml import *
from sklearn.model_selection import ParameterGrid
import random
import argparse
import sys
import shutil
import pandas as pd


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def tokenize_function(examples):
    return tokenizer(examples['sentence'], truncation=True, max_length=128)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    scmetrics.add_batch(predictions=predictions, references=labels)
    return scmetrics.compute()


def create_labels(sentiment):
    labels = []
    for s in sentiment:
        if s == 'neutral':
            labels += [0]
        elif s == 'positive':
            labels += [1]
        else:
            labels += [2]
    return labels


# parser = argparse.ArgumentParser(description='Sentence lassification task')
# parser.add_argument('--model', help='Path to pt model and tokenizer')
# config = parser.parse_args(sys.argv[1:])
task = 'sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

# Create task Dataset from annotated samples
sentences = pd.read_csv('sentiment_100.csv', header=0)
sentences = sentences[['Language', "Alissa's label"]]
dataset = Dataset.from_pandas(sentences).rename_columns({'Language': 'sentence', "Alissa's label": 'sentiment'})
dataset = dataset.add_column('label', create_labels(dataset['sentiment']))
label_dt = dataset.train_test_split(0.2)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tkn_dt = label_dt.map(tokenize_function, batched=True, num_proc=4)
# tkn_dt = tkn_dt.remove_columns([''])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=3)
if torch.cuda.is_available():
    model.to('cuda')
    print_gpu_utilization()

# Hyperparameters (for best configuration selection)
params = {
    'batch_size': [8],
    'epochs': [2],
    'learning_rate': [2e-5],
    'weight_decay': [0],
    'warmup_ratio': [0.01]
}

metrics_file = f'classification_metrics.csv'
if os.path.isfile(metrics_file):
    f = open(metrics_file, 'a')
else:
    f = open(metrics_file, 'w')
    f.write('batch_size,epochs,learning_rate,loss,f1,precision,recall\n')

best_model = []
best_precision = 0.0
tmp_trainer, tmp_comb = None, None
for comb in list(ParameterGrid(params)):
    print(f"Parameters: {comb}")
    training_args = TrainingArguments(
        output_dir=f'runs',
        evaluation_strategy='epoch',
        eval_steps=1,
        logging_strategy='epoch',
        weight_decay=comb['weight_decay'],
        warmup_ratio=comb['warmup_ratio'],
        num_train_epochs=comb['epochs'],
        learning_rate=comb['learning_rate'],
        per_device_train_batch_size=comb['batch_size'],
        per_device_eval_batch_size=comb['batch_size'],
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_precision',
        seed=42)
    scmetrics = evaluate.load("scmetrics")

    trainer = Trainer(model=model,
                      args=training_args,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
                      train_dataset=tkn_dt['train'],
                      eval_dataset=tkn_dt['test'],
                      compute_metrics=compute_metrics,
                      data_collator=data_collator)
    results = trainer.train()
    results_eval = trainer.evaluate()
    # print_summary(results)
    v = [comb['batch_size'], comb['epochs'], comb['learning_rate'], results.metrics['train_loss'],
         results_eval['eval_f1'], results_eval['eval_precision'], results_eval['eval_recall']]
    f.write(','.join([str(el) for el in v]) + '\n')

    if results_eval['eval_precision'] > best_precision:
        best_precision = results_eval['eval_precision']
        tmp_trainer = trainer
        tmp_comb = comb
    print('-' * 100)
    print('\n\n')

labels_to_sen = {0: 'neutral', 1: 'negative', 2: 'positive'}
if tmp_trainer is not None:
    best_trainer = tmp_trainer
    best_comb = tmp_comb
    print(f'Best parameters configuration: {best_comb}')
    dev_pred = best_trainer.predict(tkn_dt['test'])
    pred = np.argmax(dev_pred.predictions, axis=-1)
    pred_score = np.max(torch.nn.functional.softmax(torch.tensor(dev_pred.predictions), dim=-1).numpy(), axis=-1)
    i = 0
    errors = {'FP': [], 'FN': []}
    for pred_lab, true_lab in zip(pred, dev_pred.label_ids):
        if pred_lab != true_lab:
            if pred_lab > 1:
                errors['FP'].append((
                    tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(tkn_dt['test']['input_ids'][i])),
                    pred_score[i], labels_to_sen[pred_lab], labels_to_sen[true_lab]))
            else:
                errors['FN'].append((tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens(tkn_dt['test']['input_ids'][i])), pred_score[i],
                                     labels_to_sen[pred_lab], labels_to_sen[true_lab]))
        i += 1
    with open(f'error_analysis.tsv',
              'w') as f:
        for k, vect in errors.items():
            if k == 'FP':
                for sen in vect:
                    f.write(sen[0] + '\t' + f'PRED_{sen[2].upper()}' + '\t' + f'TRUE_{sen[3].upper()}' + '\t' + str(
                        sen[1]) + '\n')
                f.write('\n')
            else:
                for sen in vect:
                    f.write(sen[0] + '\t' + f'PRED_{sen[2].upper()}' + '\t' + f'TRUE_{sen[3].upper()}' + '\t' + str(
                        sen[1]) + '\n')
    test_pred = best_trainer.predict(tkn_dt['test'])
    print(test_pred.metrics)

    model_dir = f'runs'
    for d in os.listdir(model_dir):
        # This removes the checkpoints (comment it if you want to keep them)
        if 'checkpoint' in d:
            shutil.rmtree(os.path.join(model_dir, d))
    best_trainer.save_model(
        output_dir=f'best_model')
else:
    print("Precision is 0.0 change something in your model's configuration and retry.")
f.close()

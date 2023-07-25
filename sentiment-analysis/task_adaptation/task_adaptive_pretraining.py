from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, \
    TrainingArguments
from datasets import DatasetDict, Dataset
from pathlib import Path
from pynvml import *
import torch
import argparse
import sys


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
    return tokenizer(examples['text'], truncation=True, max_length=128)


parser = argparse.ArgumentParser(description='Task-adaptive model')
parser.add_argument('--model', help='Path to BERT-like model')
parser.add_argument('--model_name',
                    help='Name of the BERT-like model. Default = "" which corresponds to ClinicalBERT',
                    default='')
config = parser.parse_args(sys.argv[1:])

# Create task Dataset
task_dt = {'train': {},
           'dev': {},
           'test': {}}
for p in Path('/sc/arion/projects/mscic1/duplicated_content').glob('**/*.sen'):
    if 'n2c2' in str(p):
        continue
    # it = 0
    print(p)
    fold = str(p).split('/')[-1].split('.')[0]
    # with open(p, 'rb') as f:
    #     for line in f:
    #         # it += 1
    #         # if it == 101:
    #         #     break
    #         ll = line.split(b',')
    #         s = re.search(br'\xff', ll[0])
    #         if s:
    #             task_dt[fold].setdefault('text', list()).append(ll[0][:s.span()[0]].decode('utf8'))
    #         else:
    #             try:
    #                 task_dt[fold].setdefault('text', list()).append(ll[0].decode('utf8'))
    #             except UnicodeDecodeError:
    #                 continue
    with open(p, 'r') as f:
        for line in f:
            # it += 1
            # if it == 101:
            #     break
            ll = line.strip().split(',')
            if len(ll[0]) > 0:
                task_dt[fold].setdefault('text', list()).append(ll[0])

task_dt = DatasetDict({k: Dataset.from_dict(task_dt[k]) for k in task_dt.keys() if k != 'test'})
task_dt.flatten()

tokenizer = AutoTokenizer.from_pretrained(config.model)
tokenizer.add_tokens(['[DATE]', '[TIME]'], special_tokens=True)
tkn_dt = task_dt.map(tokenize_function, batched=True, num_proc=4)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

if torch.cuda.is_available():
    model = AutoModelForMaskedLM.from_pretrained(
        config.model,
        from_tf=False).to('cuda')
    print_gpu_utilization()
else:
    model = AutoModelForMaskedLM.from_pretrained(
        config.model,
        from_tf=False)
model.resize_token_embeddings(len(tokenizer))

training_args = TrainingArguments(
    output_dir=f"/sc/arion/projects/mscic1/duplicated_content/runs/ta_pretraining{config.model_name}",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    num_train_epochs=5,
    weight_decay=0.01,
    warmup_ratio=0.01,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    save_strategy='epoch',
    save_total_limit=2,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tkn_dt['train'],
    eval_dataset=tkn_dt['dev'],
    data_collator=data_collator,
)
result = trainer.train(resume_from_checkpoint=True)
print(result)
print_summary(result)

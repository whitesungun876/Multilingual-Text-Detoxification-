import os
import json
import argparse
import torch
from datasets import Dataset, concatenate_datasets
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer
)

def load_json_list(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def tokenize_examples(examples, tokenizer, lang_code):
    tokenizer.src_lang = tokenizer.tgt_lang = lang_code
    enc = tokenizer([e['source'] for e in examples], truncation=True, padding='max_length', max_length=128)
    with tokenizer.as_target_tokenizer():
        tgt_enc = tokenizer([e['target'] for e in examples], truncation=True, padding='max_length', max_length=128)
    enc['labels'] = tgt_enc['input_ids']
    return enc

class CeOnlyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            labels=inputs.get('labels')
        )
        return outputs.loss if not return_outputs else (outputs.loss, outputs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="M1 Training with CE Loss only")
    parser.add_argument('--split_dir', required=True, help='Directory with split data files')
    parser.add_argument('--output_dir', required=True, help='Directory to save model outputs')
    parser.add_argument('--langs', default='en,zh,de', help='Comma-separated list of languages')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--train_bs', type=int, default=8, help='Training batch size')
    parser.add_argument('--eval_bs', type=int, default=8, help='Evaluation batch size')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=0, help='Number of warmup steps')
    args = parser.parse_args()

    code_map = {'en': 'en_XX', 'zh': 'zh_CN', 'de': 'de_DE'}
    langs = args.langs.split(',')
    
    # Assuming the checkpoint is in a directory structure as described
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(args.split_dir)))
    ckpt = os.path.join(project_root, 'checkpoints', 'mbart_final', 'checkpoint-405')
    tokenizer = MBart50TokenizerFast.from_pretrained(ckpt, local_files_only=True)
    model = MBartForConditionalGeneration.from_pretrained(ckpt, local_files_only=True)
    model.resize_token_embeddings(len(tokenizer))

    datasets = []
    for lang in langs:
        code = code_map[lang]
        for split in ['train', 'dev']:
            path = os.path.join(args.split_dir, f'{lang}_{split}_split.json')
            examples = load_json_list(path)
            tok = tokenize_examples(examples, tokenizer, code)
            ds = Dataset.from_dict(tok).add_column('lang', [lang] * len(examples))
            datasets.append((split, ds))
    
    train_ds = concatenate_datasets([ds for s, ds in datasets if s == 'train'])
    dev_ds = concatenate_datasets([ds for s, ds in datasets if s == 'dev'])

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        save_strategy='epoch',
        save_total_limit=1,
        logging_steps=100,
        seed=42,
        report_to='none'
    )

    trainer = CeOnlyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    print("Starting training with CE Loss only.")
    trainer.train()

    final_dir = os.path.join(args.output_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("CE-only model saved to:", final_dir)
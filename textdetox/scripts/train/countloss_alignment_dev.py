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

class CountAlignTrainer(Trainer):
    def __init__(self, *args, lambda_count: float = 0.5, lambda_align: float = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_count = lambda_count
        self.lambda_align = lambda_align
        self.kl_loss = torch.nn.KLDivLoss(reduction="batchmean")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            labels=inputs.get('labels'),
            output_attentions=True
        )
        ce_loss = outputs.loss
        
        # COUNT Loss
        logits = outputs.logits
        B, T, V = logits.size()
        probs = torch.softmax(logits, dim=-1)
        flat_probs = probs.view(-1, V)
        labels_flat = inputs['labels'].view(-1)
        p_toks = flat_probs[torch.arange(flat_probs.size(0)), labels_flat].view(B, T)
        mask = (inputs.get('toxic_mask', torch.zeros_like(inputs['labels'])) == 1) & (inputs['labels'] != -100)
        count_loss = -torch.log(1 - p_toks[mask] + 1e-8).mean() if mask.any() else torch.tensor(0., device=ce_loss.device)
        
        # Alignment Loss
        attn = outputs.cross_attentions[-1].mean(dim=1)
        log_attn = torch.log(attn + 1e-8)
        uniform = torch.full_like(attn, 1.0 / attn.size(-1))
        align_loss = self.kl_loss(log_attn, uniform)
        
        loss = ce_loss + (self.lambda_count * count_loss) + (self.lambda_align * align_loss)
        return (loss, outputs) if return_outputs else loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="M1 Training with COUNT and Alignment Loss")
    parser.add_argument('--split_dir', required=True, help='Directory with split data files')
    parser.add_argument('--output_dir', required=True, help='Directory to save model outputs')
    parser.add_argument('--langs', default='en,zh,de', help='Comma-separated list of languages')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--train_bs', type=int, default=8)
    parser.add_argument('--eval_bs', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--lambda_count', type=float, default=0.5)
    parser.add_argument('--lambda_align', type=float, default=0.2)
    args = parser.parse_args()

    code_map = {'en': 'en_XX', 'zh': 'zh_CN', 'de': 'de_DE'}
    langs = args.langs.split(',')
    
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
        save_strategy='epoch',
        save_total_limit=1,
        logging_steps=100,
        seed=42,
        report_to='none'
    )

    trainer = CountAlignTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        lambda_count=args.lambda_count,
        lambda_align=args.lambda_align
    )

    print("Starting training with COUNT and Alignment Loss.")
    trainer.train()

    final_dir = os.path.join(args.output_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("COUNT+Align model saved to:", final_dir)
import os
import glob
import pickle
import argparse
from typing import List, Dict

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification
)

def load_tokenized_split(file_path: str) -> List[Dict]:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} examples from {os.path.basename(file_path)}")
    return data


def prepare_datasets(tokenized_dir: str):
    langs = ['en', 'zh', 'de']
    train_examples = []
    eval_examples  = []
    for lang in langs:
        train_path = os.path.join(tokenized_dir, f"{lang}_train_tokenized.pkl")
        dev_path   = os.path.join(tokenized_dir, f"{lang}_dev_tokenized.pkl")
        train_examples.extend(load_tokenized_split(train_path))
        eval_examples.extend(load_tokenized_split(dev_path))
    train_dataset = Dataset.from_list(train_examples)
    eval_dataset  = Dataset.from_list(eval_examples)
    print(f"Created train_dataset (size={len(train_dataset)}), eval_dataset (size={len(eval_dataset)})")
    return train_dataset, eval_dataset


class LanguageAwareToxicTrainer(Trainer):
    """
    Trainer with language-specific toxicity penalty.
    """
    def __init__(
        self,
        lambda_tox_en: float,
        lambda_tox_zh: float,
        lambda_tox_de: float,
        zh_tox_threshold: float,
        de_tox_threshold: float,
        zh_tox_model_name: str,
        multi_tox_model_name: str,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lambda_tox_en     = lambda_tox_en
        self.lambda_tox_zh     = lambda_tox_zh
        self.lambda_tox_de     = lambda_tox_de
        self.zh_tox_threshold  = zh_tox_threshold
        self.de_tox_threshold  = de_tox_threshold
        # Chinese-specific classifier
        self.zh_tox_tokenizer = AutoTokenizer.from_pretrained(zh_tox_model_name)
        self.zh_tox_model     = AutoModelForSequenceClassification.from_pretrained(
            zh_tox_model_name
        ).to(self.args.device)
        self.zh_tox_model.eval()
        # Multilingual classifier for English/German
        self.multi_tox_tokenizer = AutoTokenizer.from_pretrained(multi_tox_model_name)
        self.multi_tox_model     = AutoModelForSequenceClassification.from_pretrained(
            multi_tox_model_name
        ).to(self.args.device)
        self.multi_tox_model.eval()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Standard cross-entropy
        outputs = model(
            **{k: v for k, v in inputs.items()
               if k in ['input_ids','attention_mask','labels','decoder_attention_mask']}
        )
        ce_loss = outputs.loss

        # Generate one-best predictions for toxicity
        logits = outputs.logits
        pred_ids = torch.argmax(logits, dim=-1)
        pred_texts = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        # Compute language-aware toxicity loss
        tox_losses = []
        for text in pred_texts:
            # Determine language: CJK => Chinese, umlauts/ß => German, else English
            if any('\u4e00' <= ch <= '\u9fff' for ch in text):
                lam, thr = self.lambda_tox_zh, self.zh_tox_threshold
                tok, mod = self.zh_tox_tokenizer, self.zh_tox_model
            elif any(ch in 'äöüßÄÖÜẞ' for ch in text):
                lam, thr = self.lambda_tox_de, self.de_tox_threshold
                tok, mod = self.multi_tox_tokenizer, self.multi_tox_model
            else:
                lam, thr = self.lambda_tox_en, 0.0
                tok, mod = self.multi_tox_tokenizer, self.multi_tox_model
            enc = tok(text, return_tensors='pt', padding=True,
                      truncation=True, max_length=128).to(self.args.device)
            with torch.no_grad():
                tlogits = mod(**enc).logits
            p_tox = torch.softmax(tlogits, dim=-1)[0,1]
            # Only penalize above threshold
            loss_t = lam * torch.clamp(p_tox - thr, min=0.0)
            tox_losses.append(loss_t)
        tox_loss = torch.stack(tox_losses).mean() if tox_losses else torch.tensor(0.0, device=ce_loss.device)

        total_loss = ce_loss + tox_loss
        return (total_loss, outputs) if return_outputs else total_loss


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 (M0): mBART-50 Fine-Tuning with language-specific toxicity thresholds"
    )
    parser.add_argument("--tokenized_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda_tox_en", type=float, default=0.1,
        help="Weight for EN toxicity penalty")
    parser.add_argument("--lambda_tox_zh", type=float, default=0.2,
        help="Weight for Chinese toxicity penalty")
    parser.add_argument("--lambda_tox_de", type=float, default=0.3,
        help="Weight for German toxicity penalty")
    parser.add_argument("--zh_tox_threshold", type=float, default=0.7,
        help="Toxicity probability threshold for Chinese")
    parser.add_argument("--de_tox_threshold", type=float, default=0.5,
        help="Toxicity probability threshold for German")
    parser.add_argument(
        "--zh_tox_model_name", type=str,
        default="alibaba-pai/pai-bert-base-zh-llm-risk-detection",
        help="Chinese toxicity classifier"
    )
    parser.add_argument(
        "--multi_tox_model_name", type=str,
        default="unitary/unbiased-toxic-roberta",
        help="Multilingual toxicity classifier"
    )
    args = parser.parse_args()

    train_dataset, eval_dataset = prepare_datasets(args.tokenized_dir)

    # Model and tokenizer
    model = MBartForConditionalGeneration.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt"
    )
    tokenizer = MBart50TokenizerFast.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt"
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding="longest"
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = LanguageAwareToxicTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        lambda_tox_en=args.lambda_tox_en,
        lambda_tox_zh=args.lambda_tox_zh,
        lambda_tox_de=args.lambda_tox_de,
        zh_tox_threshold=args.zh_tox_threshold,
        de_tox_threshold=args.de_tox_threshold,
        zh_tox_model_name=args.zh_tox_model_name,
        multi_tox_model_name=args.multi_tox_model_name
    )

    print(f"Training size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    print(f"λ_en={args.lambda_tox_en}, λ_zh={args.lambda_tox_zh}, λ_de={args.lambda_tox_de}",
          f"thr_zh={args.zh_tox_threshold}, thr_de={args.de_tox_threshold}\n")

    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("Training completed. Model saved to final/")

if __name__ == "__main__":
    main()


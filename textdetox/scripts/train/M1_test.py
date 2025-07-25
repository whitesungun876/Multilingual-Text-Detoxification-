import os
import json
import argparse
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from datasets import Dataset, concatenate_datasets
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    GPT2LMHeadModel,
    GPT2TokenizerFast
)
import sacrebleu
from bert_score import BERTScorer
import numpy as np

# --- Joint computation components: STA × SIM × Fluency ---
# These components measure different aspects of translation quality.
_bert_scorer = BERTScorer(
    model_type="bert-base-multilingual-cased",
    lang="en",
    rescale_with_baseline=True
)
_lm_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
_lm_model     = GPT2LMHeadModel.from_pretrained("gpt2").eval()

def semantic_alignment(ref: str, hyp: str) -> float:
    """Computes BERTScore F1 for semantic alignment."""
    P, R, F1 = _bert_scorer.score([hyp], [ref])
    return F1[0].item()

def reference_similarity(ref: str, hyp: str) -> float:
    """Computes BLEU score for similarity."""
    bleu = sacrebleu.sentence_bleu(hyp, [ref]).score
    return bleu / 100.0

def fluency_score(text: str) -> float:
    """Calculates fluency using a GPT-2 model."""
    if not text: return 0.0
    enc = _lm_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        loss = _lm_model(**enc, labels=enc["input_ids"]).loss
    return torch.exp(-loss).item()

def compute_joint_score(hyp: str, ref: str) -> float:
    """Computes the joint quality score (STA * SIM * Fluency)."""
    return semantic_alignment(ref, hyp) * reference_similarity(ref, hyp) * fluency_score(hyp)

class PairwiseRankingToxicTrainer(Trainer):
    """
    Trainer that combines:
    1. Cross-Entropy Loss (MLE)
    2. Token-Level Count Unlikelihood Loss
    3. Alignment Regularization
    4. Pairwise Ranking Loss
    """
    def __init__(
        self,
        lambda_count: float = 0.5,
        lambda_align: float = 0.2,
        lambda_rank: float = 0.5,
        ranking_margin: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.lambda_count = lambda_count
        self.lambda_align = lambda_align
        self.lambda_rank = lambda_rank
        self.ranking_margin = ranking_margin
        self.kl_loss = torch.nn.KLDivLoss(reduction="batchmean")

    def compute_loss(self, model, inputs, return_outputs=False):
        # --- 1. Standard Cross-Entropy Loss ---
        labels = inputs.pop("labels")
        # Pop reference_texts before passing to model
        reference_texts = inputs.pop("reference_texts")
        
        outputs = model(
            **inputs,
            labels=labels,
            output_attentions=True
        )
        ce_loss = outputs.loss

        # --- 2. Token-Level Count Loss ---
        logits = outputs.logits
        B, T, V = logits.size()
        probs = torch.softmax(logits, dim=-1)
        
        # Get probabilities of target tokens
        flat_probs = probs.view(-1, V)
        labels_flat = labels.view(-1)
        p_toks = flat_probs[torch.arange(flat_probs.size(0)), labels_flat].view(B, T)

        # Penalize toxic tokens based on a mask
        mask = (inputs.get('toxic_mask', torch.zeros_like(labels)) == 1) & (labels != -100)
        count_loss = -torch.log(1 - p_toks[mask] + 1e-9).mean() if mask.any() else torch.tensor(0., device=ce_loss.device)

        # --- 3. Alignment Regularization Loss ---
        # Encourages attention to be more uniform, preventing over-reliance on specific source tokens
        cross_attentions = outputs.cross_attentions[-1]
        mean_attention = cross_attentions.mean(dim=1) # Average over heads
        log_attention = torch.log(mean_attention + 1e-9)
        uniform_dist = torch.full_like(mean_attention, 1.0 / mean_attention.size(-1))
        align_loss = self.kl_loss(log_attention, uniform_dist)
        
        # --- 4. Pairwise Ranking Loss ---
        # Generate model's hypothesis
        pred_ids = torch.argmax(logits, dim=-1)
        hyp_texts = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        
        # Compute joint quality scores for hypothesis and reference
        # The reference is the "positive" example, the hypothesis is the "negative"
        scores_hyp = [compute_joint_score(hyp, ref) for hyp, ref in zip(hyp_texts, reference_texts)]
        scores_ref = [compute_joint_score(ref, ref) for ref in reference_texts] # Score of reference against itself
        
        scores_hyp_t = torch.tensor(scores_hyp, device=ce_loss.device)
        scores_ref_t = torch.tensor(scores_ref, device=ce_loss.device)
        
        # Hinge loss: penalize if score(hyp) is not lower than score(ref) by a margin
        ranking_loss = torch.clamp(self.ranking_margin - (scores_ref_t - scores_hyp_t), min=0).mean()

        # --- Combine all losses ---
        total_loss = ce_loss + (self.lambda_count * count_loss) + \
                     (self.lambda_align * align_loss) + \
                     (self.lambda_rank * ranking_loss)

        return (total_loss, outputs) if return_outputs else total_loss

def load_json_list(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def tokenize_and_prepare_features(examples, tokenizer, lang_code):
    """Tokenizes source and target, and keeps reference text for loss calculation."""
    tokenizer.src_lang = tokenizer.tgt_lang = lang_code
    
    # Standard tokenization
    source_texts = [e['source'] for e in examples]
    target_texts = [e['target'] for e in examples]
    
    enc = tokenizer(source_texts, truncation=True, padding='max_length', max_length=128)
    with tokenizer.as_target_tokenizer():
        tgt_enc = tokenizer(target_texts, truncation=True, padding='max_length', max_length=128)
    
    enc['labels'] = tgt_enc['input_ids']
    # Add raw reference text for use in pairwise ranking loss
    enc['reference_texts'] = target_texts
    
    return enc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train M1 with Count, Align, and Pairwise Ranking Loss")
    parser.add_argument('--split_dir', required=True, help="Directory with train/dev splits")
    parser.add_argument('--output_dir', required=True, help="Directory to save model outputs")
    parser.add_argument('--langs', default='en,zh,de', help='Comma-separated languages')
    parser.add_argument('--epochs', type=int, default=3, help="Number of training epochs")
    parser.add_argument('--train_bs', type=int, default=8, help="Training batch size")
    parser.add_argument('--eval_bs', type=int, default=8, help="Evaluation batch size")
    parser.add_argument('--lr', type=float, default=3e-5, help="Learning rate")
    
    # Loss weights
    parser.add_argument('--lambda_count', type=float, default=0.5, help="Weight for count loss")
    parser.add_argument('--lambda_align', type=float, default=0.2, help='Weight for alignment loss')
    parser.add_argument('--lambda_rank', type=float, default=0.5, help="Weight for pairwise ranking loss")
    parser.add_argument('--ranking_margin', type=float, default=0.1, help="Margin for pairwise ranking loss")
    args = parser.parse_args()

    code_map = {'en': 'en_XX', 'zh': 'zh_CN', 'de': 'de_DE'}
    langs = args.langs.split(',')
    
    # Assume checkpoint is in a parent directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(args.split_dir)))
    ckpt = os.path.join(project_root, 'checkpoints', 'mbart_final', 'checkpoint-405')
    tokenizer = MBart50TokenizerFast.from_pretrained(ckpt, local_files_only=True)
    model = MBartForConditionalGeneration.from_pretrained(ckpt, local_files_only=True)
    model.resize_token_embeddings(len(tokenizer))

    # Load and process datasets
    datasets = []
    for lang in langs:
        code = code_map[lang]
        for split in ['train', 'dev']:
            path = os.path.join(args.split_dir, f'{lang}_{split}_split.json')
            examples = load_json_list(path)
            # Use the new tokenization function
            processed_features = tokenize_and_prepare_features(examples, tokenizer, code)
            ds = Dataset.from_dict(processed_features)
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
        report_to="none"
    )

    trainer = PairwiseRankingToxicTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        lambda_count=args.lambda_count,
        lambda_align=args.lambda_align,
        lambda_rank=args.lambda_rank,
        ranking_margin=args.ranking_margin
    )

    print("Starting training with Count + Align + Pairwise Ranking Loss.")
    print(f"Loss weights: Count(λ={args.lambda_count}), Align(λ={args.lambda_align}), Rank(λ={args.lambda_rank})")
    
    trainer.train()

    final_dir = os.path.join(args.output_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("Training complete. Model saved to:", final_dir)
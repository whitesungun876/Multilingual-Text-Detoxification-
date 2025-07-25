import os
import json
import sys
import argparse

# Enable CUDA synchronous error reporting for easier debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch

# Verify dependencies
try:
    from sentence_transformers import SentenceTransformer, util as st_util
except ImportError:
    print("ERROR: sentence-transformers not installed. Please install via 'pip install sentence-transformers'.")
    sys.exit(1)
try:
    import evaluate
except ImportError:
    print("ERROR: evaluate not installed. Please install via 'pip install evaluate'.")
    sys.exit(1)
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    AutoModelForSequenceClassification,
    AutoTokenizer
)

# Define device globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_json_splits(splits_dir: str, langs: list):
    data = {}
    for lang in langs:
        path = os.path.join(splits_dir, f"{lang}_dev_split.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing split file: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            entries = json.load(f)
        srcs = [e['source'] for e in entries]
        refs = [e['target'] for e in entries]
        data[lang] = {'src': srcs, 'ref': refs}
    return data


def setup_mbart(model_dir: str):
    model = MBartForConditionalGeneration.from_pretrained(model_dir).to(device)
    tokenizer = MBart50TokenizerFast.from_pretrained(model_dir)
    return model, tokenizer


def generate_mbart_batch(model, tokenizer, texts, lang, max_length=128, num_beams=5):
    code = {'en': 'en_XX', 'zh': 'zh_CN', 'de': 'de_DE'}[lang]
    tokenizer.src_lang = code
    tokenizer.tgt_lang = code
    batch = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)
    out = model.generate(
        **batch,
        forced_bos_token_id=tokenizer.lang_code_to_id[code],
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True
    )
    return tokenizer.batch_decode(out, skip_special_tokens=True)


def compute_metrics(srcs, preds, refs,
                    tox_tok, tox_mod,
                    sim_mod, comet_metric,
                    sta_batch=32, sim_batch=32):
    # 1) STA: binary non-toxic accuracy
    STA_bits = []
    for i in range(0, len(preds), sta_batch):
        texts = preds[i:i+sta_batch]
        enc = tox_tok(texts, return_tensors='pt', padding=True,
                      truncation=True, max_length=128).to(device)
        with torch.no_grad():
            logits = tox_mod(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, 0]
        STA_bits.extend((probs >= 0.5).long().cpu().tolist())
    STA = sum(STA_bits) / len(STA_bits)

    # 2) SIM: LaBSE cosine(src, pred)
    SIM_scores = []
    for i in range(0, len(srcs), sim_batch):
        batch_src = srcs[i:i+sim_batch]
        batch_prd = preds[i:i+sim_batch]
        emb_src = sim_mod.encode(batch_src, convert_to_tensor=True, device=device)
        emb_prd = sim_mod.encode(batch_prd, convert_to_tensor=True, device=device)
        sims = st_util.pytorch_cos_sim(emb_src, emb_prd).diag()
        SIM_scores.extend(sims.cpu().tolist())
    SIM = sum(SIM_scores) / len(SIM_scores)

    # 3) Fluency: COMET-DA scores
    comet_res = comet_metric.compute(sources=srcs, predictions=preds, references=refs)
    FLU = comet_res['scores']
    Flu = sum(FLU) / len(FLU)

    # 4) Joint
    JOINT_scores = [s * m * f for s, m, f in zip(STA_bits, SIM_scores, FLU)]
    Joint = sum(JOINT_scores) / len(JOINT_scores)

    return {'STA': STA, 'SIM': SIM, 'Flu': Flu, 'Joint': Joint}


def evaluate_m0(m0_dir, splits_dir, langs):
    # Load splits and MBART
    data = load_json_splits(splits_dir, langs)
    model, tokenizer = setup_mbart(m0_dir)

    # Load proxies once
    tox_tok = AutoTokenizer.from_pretrained('unitary/unbiased-toxic-roberta')
    tox_mod = AutoModelForSequenceClassification.from_pretrained(
        'unitary/unbiased-toxic-roberta'
    ).to(device).eval()
    sim_mod = SentenceTransformer('sentence-transformers/LaBSE', device=device)
    comet_metric = evaluate.load('comet', model_id='unbabel/wmt20-comet-da', module_type='metric')

    print('| Lang | STA   | SIM   | Flu   | Joint |')
    print('|------|-------|-------|-------|-------|')
    for lang in langs:
        srcs = data[lang]['src']
        refs = data[lang]['ref']
        preds = []
        for i in range(0, len(srcs), 16):
            batch_src = srcs[i:i+16]
            preds.extend(generate_mbart_batch(model, tokenizer, batch_src, lang))
        metrics = compute_metrics(
            srcs, preds, refs,
            tox_tok, tox_mod,
            sim_mod, comet_metric
        )
        print(f"| {lang:<4} | {metrics['STA']:.4f} | {metrics['SIM']:.4f} | "
              f"{metrics['Flu']:.4f} | {metrics['Joint']:.4f} |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m0_model_dir', required=True)
    parser.add_argument('--splits_dir',   required=True)
    args = parser.parse_args()

    evaluate_m0(
        args.m0_model_dir,
        args.splits_dir,
        langs=['en', 'zh', 'de']
    )

if __name__ == '__main__':
    main()


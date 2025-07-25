import os
import json
import argparse
import pickle
from transformers import MBart50TokenizerFast
from tqdm import tqdm

def tokenize_and_save(lang_code, input_path, out_dir):
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    if lang_code == "en":
        tokenizer.src_lang = "en_XX"
        tokenizer.tgt_lang = "en_XX"
    elif lang_code == "de":
        tokenizer.src_lang = "de_DE"
        tokenizer.tgt_lang = "de_DE"
    elif lang_code == "zh":
        tokenizer.src_lang = "zh_CN"
        tokenizer.tgt_lang = "zh_CN"

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)  
    print(f"Loaded {len(data)} examples to tokenize for {lang_code}")

    tokenized = []
    for item in tqdm(data, desc=f"Tokenizing {lang_code}"):
        src_text = item["source"]
        tgt_text = item["target"]
        # encode source → input_ids + attention_mask
        encoded_src = tokenizer(src_text, 
                                max_length=128, 
                                padding="max_length", 
                                truncation=True, 
                                return_tensors="pt")
        # encode target → labels (shifted inside Trainer)
        encoded_tgt = tokenizer(tgt_text, 
                                max_length=128, 
                                padding="max_length", 
                                truncation=True, 
                                return_tensors="pt")

        tokenized.append({
            "input_ids":      encoded_src["input_ids"][0].tolist(),
            "attention_mask": encoded_src["attention_mask"][0].tolist(),
            "labels":         encoded_tgt["input_ids"][0].tolist()
        })

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{lang_code}_tokenized.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(tokenized, f)
    print(f"Saved tokenized data for {lang_code} ({len(tokenized)} examples) to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang",     choices=["en","zh","de"], required=True)
    parser.add_argument("--input",    type=str, required=True,
                        help="formalized JSON file path，例如 data/processed/normalized/en_all_norm.json")
    parser.add_argument("--output",   type=str, required=True,
                        help="output catalog，save tokenized.pkl ")
    args = parser.parse_args()

    tokenize_and_save(args.lang, args.input, args.output)

if __name__ == "__main__":
    main()

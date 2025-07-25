import os
import json
import traceback
from datasets import load_dataset

def save_language(lang_code, out_dir):
    try:
        print(f"\n--> Start processing language: {lang_code}")

        print("    Loading `textdetox/multilingual_paradetox` …")
        ds_dict = load_dataset("textdetox/multilingual_paradetox")
        print(f"    Available keys in dataset dict: {list(ds_dict.keys())}")

        if lang_code not in ds_dict:
            print(f"    ERROR: language '{lang_code}' not found in dataset keys → {list(ds_dict.keys())}")
            return

        ds_lang = ds_dict[lang_code]
        count = len(ds_lang)
        print(f"    (1) Found {count} examples for '{lang_code}'")

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{lang_code}_all.json")

        examples = []
        for idx, item in enumerate(ds_lang):
          
            if "toxic_sentence" not in item or "neutral_sentence" not in item:
                print(f"    WARNING: 'toxic_sentence' or 'neutral_sentence' missing at index {idx} for lang {lang_code}")
                continue
            src = item["toxic_sentence"]
            tgt = item["neutral_sentence"]
            examples.append({"source": src, "target": tgt})

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        print(f"    (2) Saved {len(examples)} items to {out_path}")

    except Exception as e:
        print(f"    Exception occurred while processing {lang_code}: {e}")
        traceback.print_exc()

def main():
    langs = ["en", "zh", "de"]
    out_base = "data/raw/CLEF_parallel"

    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")

    for lang in langs:
        save_language(lang, out_base)

if __name__ == "__main__":
    main()

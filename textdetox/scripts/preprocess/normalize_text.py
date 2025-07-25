
import os
import re
import json
import argparse
import jieba

def normalize_en_de(text: str) -> str:
    
    text = text.lower()
    punctuation_map = {
        "，": ",", "。": ".", "！": "!", "？": "?", "；": ";", "：": ":",
        "“": '"', "”": '"', "‘": "'", "’": "'", "（": "(", "）": ")"
    }
    for zh_p, en_p in punctuation_map.items():
        text = text.replace(zh_p, en_p)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_zh(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "", text)
    punctuation_map = {
        ",": "，", 
        r"\.": "。", r"!": "！", r"\?": "？",
    }
    for pattern, repl in punctuation_map.items():
        text = re.sub(pattern, repl, text)
    return text

def normalize_file(input_path: str, output_path: str, lang: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    normalized = []
    for item in data:
        src = item["source"]
        tgt = item["target"]
        if lang in ["en", "de"]:
            src_norm = normalize_en_de(src)
            tgt_norm = normalize_en_de(tgt)
        elif lang == "zh":
            src_norm = normalize_zh(src)
            tgt_norm = normalize_zh(tgt)
        else:
            src_norm, tgt_norm = src, tgt
        normalized.append({"source": src_norm, "target": tgt_norm})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)
    print(f"Normalized {len(data)} examples for {lang} → saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang",      choices=["en","zh","de"], required=True)
    parser.add_argument("--input",     type=str, required=True,
                        help="raw JSON file pathway（*_all.json or *_train.json）")
    parser.add_argument("--output",    type=str, required=True,
                        help="normalized JSON file pathway")
    args = parser.parse_args()

    normalize_file(args.input, args.output, args.lang)

if __name__ == "__main__":
    main()

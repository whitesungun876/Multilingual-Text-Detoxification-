import os
import json
import argparse
from typing import List, Dict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.model_selection import train_test_split

# --- Helper Functions ---

def load_json_data(file_path: str) -> List[Dict]:
    """Loads a list of dictionaries from a JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_data(data: List[Dict], file_path: str):
    """Saves a list of dictionaries to a JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(data)} pairs to {file_path}")

def load_toxic_lexicon(lang: str) -> List[str]:
    """Loads a predefined list of toxic words for a given language."""
    # This is a placeholder. You should replace this with your actual toxic lexicons.
    lexicons = {
        "de": ["schlecht", "böse", "giftig", "toxisch"],
        "zh": ["坏", "有毒", "不好"]
    }
    if lang not in lexicons:
        print(f"Warning: No toxic lexicon found for '{lang}'. Filtering will be skipped.")
        return []
    return lexicons[lang]

def contains_toxic_words(text: str, lexicon: List[str]) -> bool:
    """Checks if a text contains any word from the toxic lexicon."""
    return any(toxic_word in text for toxic_word in lexicon)

# --- Main Augmentation Logic ---

class BackTranslator:
    """A class to handle the back-translation and detoxification process."""
    def __init__(self, target_lang: str, batch_size: int = 16):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.target_lang = target_lang

        print(f"Using device: {self.device}")
        print("Loading translation and detoxification models...")

        # 1. English -> Target Language Model
        en_to_target_model_name = f"Helsinki-NLP/opus-mt-en-{target_lang}"
        self.en_to_target_tokenizer = AutoTokenizer.from_pretrained(en_to_target_model_name)
        self.en_to_target_model = AutoModelForSeq2SeqLM.from_pretrained(en_to_target_model_name).to(self.device)

        # 2. English Detoxification Model (Placeholder)
        # Replace this with your actual English detoxification model.
        # For this example, we use a simple EN->EN model as a stand-in.
        detox_model_name = "t5-small"
        self.detox_tokenizer = AutoTokenizer.from_pretrained(detox_model_name)
        self.detox_model = AutoModelForSeq2SeqLM.from_pretrained(detox_model_name).to(self.device)

        # 3. Target Language -> English Model
        target_to_en_model_name = f"Helsinki-NLP/opus-mt-{target_lang}-en"
        self.target_to_en_tokenizer = AutoTokenizer.from_pretrained(target_to_en_model_name)
        self.target_to_en_model = AutoModelForSeq2SeqLM.from_pretrained(target_to_en_model_name).to(self.device)
        
        print("Models loaded successfully.")

    def translate(self, texts: List[str], model, tokenizer, prompt: str = "") -> List[str]:
        """Generic translation function."""
        all_translations = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc=f"Translating"):
            batch = [prompt + text for text in texts[i:i+self.batch_size]]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
            translated_ids = model.generate(**inputs, max_length=128)
            translations = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
            all_translations.extend(translations)
        return all_translations

    def run_augmentation(self, en_toxic_data: List[Dict]) -> List[Dict]:
        """Executes the full back-translation augmentation pipeline."""
        
        # Extract original English toxic source texts
        en_toxic_texts = [item['source'] for item in en_toxic_data]

        # Step 1: Translate English toxic text -> target language (New Source)
        print("Step 1: Translating English to target language...")
        new_source_texts = self.translate(en_toxic_texts, self.en_to_target_model, self.en_to_target_tokenizer)

        # Step 2: Detoxify the original English toxic text
        print("Step 2: Detoxifying English text...")
        detox_prompt = "detoxify: " # Example prompt for a T5-style model
        detoxified_en_texts = self.translate(en_toxic_texts, self.detox_model, self.detox_tokenizer, prompt=detox_prompt)

        # Step 3: Back-translate the detoxified English -> target language (New Target)
        print("Step 3: Back-translating detoxified English to target language...")
        new_target_texts = self.translate(detoxified_en_texts, self.en_to_target_model, self.en_to_target_tokenizer)

        # Step 4: Filter pairs based on a toxic lexicon
        print("Step 4: Filtering results...")
        toxic_lexicon = load_toxic_lexicon(self.target_lang)
        augmented_pairs = []
        for src, tgt in zip(new_source_texts, new_target_texts):
            if not contains_toxic_words(tgt, toxic_lexicon):
                augmented_pairs.append({"source": src, "target": tgt})
        
        print(f"Filtered out {len(new_source_texts) - len(augmented_pairs)} pairs containing toxic lexicon.")
        print(f"Generated {len(augmented_pairs)} clean augmented pairs.")
        
        return augmented_pairs


def main():
    parser = argparse.ArgumentParser(description="Augment data using back-translation for detoxification.")
    parser.add_argument("--lang", choices=["zh", "de"], required=True, 
                        help="Target language for augmentation.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the JSON file containing English toxic data (e.g., data/en_toxic.json).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the augmented train and dev files.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for the train/dev split.")
    args = parser.parse_args()

    # Load initial data
    en_data = load_json_data(args.input_file)

    # Perform augmentation
    translator = BackTranslator(target_lang=args.lang)
    augmented_data = translator.run_augmentation(en_data)

    if not augmented_data:
        print("No data was generated after filtering. Exiting.")
        return

    # Split data into 90% train and 10% dev
    train_data, dev_data = train_test_split(
        augmented_data,
        test_size=0.1,
        random_state=args.seed
    )

    print(f"Splitting data: {len(train_data)} for train, {len(dev_data)} for dev.")

    # Save the final files
    train_path = os.path.join(args.output_dir, f"{args.lang}_train_augmented.json")
    dev_path = os.path.join(args.output_dir, f"{args.lang}_dev_augmented.json")
    
    save_json_data(train_data, train_path)
    save_json_data(dev_data, dev_path)

    print("Augmentation process completed successfully.")


if __name__ == "__main__":
    main()
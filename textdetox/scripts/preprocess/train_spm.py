# train_spm.py
import sentencepiece as spm

def train_unigram(input_file, model_prefix, vocab_size=3000, char_coverage=0.9995):
    spm.SentencePieceTrainer.Train(
        input='zh_detox_corpus.txt',
        model_prefix='zh_unigram',
        vocab_size=32000,           
        model_type='unigram',
        character_coverage=0.9995,
        hard_vocab_limit=False 
    )
    print(f"Trained SentencePiece model '{model_prefix}.model' and '{model_prefix}.vocab'")

if __name__ == "__main__":
    train_unigram(
        input_file="zh_detox_corpus.txt",
        model_prefix="zh_unigram",
        vocab_size=32000,
        char_coverage=0.9995
    )

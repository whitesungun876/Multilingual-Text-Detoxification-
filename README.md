# Multilingual-Text-Detoxification
## Overview

This project implements a unified, two-stage multilingual text detoxification framework for English, Chinese, and German. It rewrites toxic content (harassment, profanity, hate speech) into non-toxic equivalents while preserving the original meaning, promoting healthier online discourse.

## Key Features

* **Two-Stage Pipeline**

  1. **Stage 1 (M0)**: Fine-tune `facebook/mbart-large-50-many-to-many-mmt` using standard cross-entropy on parallel "toxic → non-toxic" corpora with script-aware toxicity penalties.
  2. **Stage 2 (M1)**: Joint end-to-end training with four losses:

     * Cross-Entropy Loss
     * Contrastive Unlikelihood Loss (penalizes copying toxic tokens)
     * Alignment Regularization Loss (aligns decoder attention to reference alignments)
     * Pairwise Ranking Loss (directly optimizes the joint detoxification metric)

* **Script-Aware Penalties**: Custom token-level toxicity penalties for CJK scripts and German compounds.

* **Back-Translation Augmentation**: Generate pseudo-parallel data for Chinese and German via English back-translation and filter residual toxic tokens.

* **Comprehensive Evaluation**: Demonstrated significant gains on CLEF 2025 dev and test sets across STA, SIM, Fluency, and joint metrics.

## Dependencies

* Python 3.8+
* PyTorch 1.10+
* Transformers 4.30+
* sentencepiece 0.1.96
* jieba 0.42.1
* numpy, scipy, scikit-learn

Install with:

```bash
pip install torch transformers sentencepiece jieba numpy scipy scikit-learn
```

## Project Structure

The repository is organized as follows:

```
├── scripts/                # Training and evaluation scripts
│   ├── CE_test.py          # Stage 1 (M0) test script
│   ├── M1_test.py          # Stage 2 (M1) test script
│   ├── CE_dev.py           # Stage 1 (M0) development evaluation
│   ├── M1_dev.py           # Stage 2 (M1) development evaluation
│   ├── countloss_dev.py    # Contrastive Unlikelihood loss dev evaluation
│   ├── countloss_alignment_dev.py  # Count + Alignment dev evaluation
│   └── backtranslate.sh    # Data augmentation script (optional)
├── evaluate/               # Standalone evaluation scripts
│   └── evaluate1.py        # Comprehensive metrics evaluation
├── data/                   # Dataset directories
│   ├── raw/                # Original CLEF 2025 parallel corpora
│   ├── processed/          # Preprocessed data (tokenized, normalized)
│   ├── final/              # Final training/dev/test splits
│   ├── splits/             # Train/dev/test indexes and metadata
│   └── lexicons/           # Toxicity lexicons and script-specific resources
├── outputs/                # Model checkpoints and results
│   ├── m0/                 # Stage 1 outputs
│   └── m1/                 # Stage 2 outputs
├── train_stage1.py         # Entry point for Stage 1 training
├── train_stage2.py         # Entry point for Stage 2 training
├── infer.py                # Inference script
├── evaluate.py             # Shortcut for evaluation and metrics
├── README.md               # Project overview and instructions
└── LICENSE                 # License information
```

## Data Preparation

1. Download CLEF 2025 parallel "toxic → non-toxic" datasets for English, Chinese, and German.
2. Organize under `data/`:

   ```
   data/
   ├── en-toxic.txt
   ├── en-clean.txt
   ├── zh-toxic.txt
   ├── zh-clean.txt
   ├── de-toxic.txt
   └── de-clean.txt
   ```
3. (Optional) Run `scripts/backtranslate.sh` to generate back-translated Chinese and German data.

## Training

### Stage 1: Baseline Fine-Tuning (M0)

```bash
python train_stage1.py \
  --model_name facebook/mbart-large-50-many-to-many-mmt \
  --train_data_dir data/ \
  --output_dir outputs/m0 \
  --epochs 3 --batch_size 8 --lr 3e-5
```

### Stage 2: Joint Multi-Loss Training (M1)

```bash
python train_stage2.py \
  --init_model outputs/m0/checkpoint-best \
  --train_data_dir data/ \
  --output_dir outputs/m1 \
  --alpha 0.2 --beta 0.5 --gamma 0.05 \
  --epochs 5 --batch_size 8 --lr 3e-5
```

## Evaluation and Inference

```bash
python evaluate.py \
  --model_dir outputs/m1 \
  --test_data_dir data/ \
  --metrics STA SIM Fluency
```

Inference example:

```bash
python infer.py \
  --model_dir outputs/m1 \
  --input_file examples/input.txt \
  --output_file examples/output.txt
```

## Results

| Language | Model | STA   | SIM   | Fluency | Joint |
| -------- | ----- | ----- | ----- | ------- | ----- |
| English  | M0    | 0.450 | 0.809 | 0.809   | 0.295 |
| English  | M1    | 0.460 | 0.790 | 0.810   | 0.295 |
| Chinese  | M0    | 0.250 | 0.818 | 0.775   | 0.159 |
| Chinese  | M1    | 0.261 | 0.828 | 0.808   | 0.175 |
| German   | M0    | 0.390 | 0.911 | 0.798   | 0.283 |
| German   | M1    | 0.420 | 0.904 | 0.870   | 0.330 |

## Citation

```bibtex
@inproceedings{Lian2025Multilingual,
  title={Multilingual Text Detoxification for English, Chinese, and German},
  author={Lian, Jieyu and others},
  booktitle={CLEF 2025},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.



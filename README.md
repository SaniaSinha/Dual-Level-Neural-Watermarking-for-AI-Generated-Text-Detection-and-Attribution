# Dual-Level Neural Watermarking for AI-Generated Text Detection and Attribution

This project detects AI-generated text and embeds a source-aware watermark for attribution and ownership verification.

## Features

- Fine-tuned DistilBERT AI-text detector
- 16-bit source-aware watermark
- 4-bit source attribution ID
- 12-bit SHA-256 verification signature
- Level 1 structure phrase watermarking
- Level 2 semantic marker watermarking
- Source recovery and ownership verification
- Tamper-sensitive decoding

## Run on Kaggle

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train detector

```bash
python detector_train.py
```

### Run demo

```bash
python demo.py
```

### Interactive testing

```bash
python interactive_test.py
```

## Watermark Format

16-bit watermark:

```
4-bit Source ID + 12-bit Verification Signature
```

Example:

```
0101 001100110011
```

Where:

- `0101` = Student Custom Model
- `001100110011` = SHA-256 verification bits

> **Important:** For normal unwatermarked text, source attribution is only probabilistic. Reliable source attribution is possible only when the text is generated or processed through this watermarking pipeline.

## Kaggle Run Order

```python
!pip install -q transformers datasets accelerate sentencepiece scikit-learn matplotlib pandas numpy
!python detector_train.py
!python demo.py
!python interactive_test.py
```

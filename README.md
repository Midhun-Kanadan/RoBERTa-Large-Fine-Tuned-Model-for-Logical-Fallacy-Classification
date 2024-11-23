# RoBERTa-Large-Fine-Tuned-Model-for-Logical-Fallacy-Classification

This repository contains the code to fine-tune a `roberta-large` model for logical fallacy detection using the [Logical Fallacy Dataset](https://huggingface.co/datasets/MidhunKanadan/logical-fallacy-classification). The model is trained to classify 13 different types of logical fallacies in text.

## Training Parameters
- **Base Model**: `roberta-large`
- **Dataset**: Logical Fallacy Dataset
- **Number of Classes**: 13
- **Training Parameters**:
  - **Learning Rate**: 2e-6 (linear scheduler)
  - **Batch Size**: 8 (gradient accumulation for effective batch size of 16)
  - **Weight Decay**: 0.01
  - **Training Epochs**: 15
  - **Mixed Precision (FP16)**: Enabled
  - **Evaluation Strategy**: Per epoch
  - **Metric for Best Model**: Weighted F1-score
  - **Save Strategy**: Save best model per epoch

## Features
- Handles imbalanced datasets with weighted loss computation.
- Trained with Hugging Face `transformers` library for seamless integration.
- Saves the model, tokenizer, and label mapping for easy reuse.

## How to Use
Run the training script directly to reproduce the model:
```bash
python finetuning_logical_fallacy_model.py
```
The trained model, tokenizer, and label mappings will be saved in the ./results directory.
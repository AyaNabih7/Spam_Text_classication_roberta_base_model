# Spam Text Classification — RoBERTa Base

A reproducible Jupyter Notebook workflow that trains a RoBERTa-base model to classify text as spam or not spam. The repository contains a single notebook (Spam_Text_classication_roberta_base_model.ipynb) with data preprocessing, tokenization using Hugging Face Transformers, model training (PyTorch), evaluation, and model saving steps.

## Table of contents
- Overview
- Notebook summary
- Requirements
- Quick start
  - Run in Google Colab
  - Run locally
- Dataset
- Model & training
- Outputs
- Reproduce results
- File structure
- Contributing
- License
- Contact

## Overview
This project demonstrates how to fine-tune a RoBERTa-base model for binary spam detection. The notebook walks through:
- Loading and exploring the dataset
- Text preprocessing and label encoding
- Tokenization with RoBERTa tokenizer
- Creating PyTorch datasets and dataloaders
- Fine-tuning RoBERTa for classification
- Evaluating the model (accuracy, precision, recall, F1, confusion matrix)
- Saving the trained model and tokenizer

## Notebook summary
Open `Spam_Text_classication_roberta_base_model.ipynb` to see the full pipeline. High-level sections:
1. Environment setup (install packages, GPU settings)
2. Data loading & exploration
3. Preprocessing and splitting (train/val/test)
4. Tokenization (Hugging Face `RobertaTokenizer`)
5. Model creation (`RobertaForSequenceClassification` or equivalent)
6. Training loop (optimizer, scheduler, loss)
7. Evaluation and metrics (classification report & confusion matrix)
8. Saving/loading the model & tokenizer

## Requirements
- Python 3.8+
- PyTorch (compatible with your CUDA version if using GPU)
- transformers (Hugging Face)
- datasets (optional)
- scikit-learn
- pandas
- numpy
- tqdm
- matplotlib / seaborn (for plots)

Install the most common dependencies:
pip install torch transformers datasets scikit-learn pandas numpy tqdm matplotlib seaborn

Note: The notebook may include inline pip installs for Colab. Check the first cells of the notebook for exact versions used.

## Quick start

### Run in Google Colab
1. Open the notebook in Colab: use the GitHub file view → "Open in Colab" (or upload the notebook to your Colab workspace).
2. Run all cells. Colab provides GPU runtime — set Runtime > Change runtime type > GPU.
3. Follow any prompts to mount Drive or download datasets if the notebook downloads them.

### Run locally
1. Clone this repository:
   git clone https://github.com/AyaNabih7/Spam_Text_classication_roberta_base_model.git
2. Create and activate a virtual environment:
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
3. Install dependencies:
   pip install -r requirements.txt
   (If no requirements.txt included, use the install command in Requirements above.)
4. Launch Jupyter Notebook / JupyterLab and open `Spam_Text_classication_roberta_base_model.ipynb`:
   jupyter lab
5. Run cells sequentially.

## Dataset
The notebook expects a text dataset containing messages and labels (spam/ham). Typical layout:
- CSV with columns: `text` (message content), `label` (0/1 or 'spam'/'ham')

If the notebook downloads a public dataset (e.g., SMS Spam Collection), it will run that automatically. Otherwise, update the data-loading cell to point to your dataset file path.

## Model & training
- Model: RoBERTa-base from Hugging Face (`roberta-base`) fine-tuned for binary classification
- Tokenizer: `RobertaTokenizer` (with max sequence length set in the notebook)
- Typical hyperparameters (check notebook for exact values):
  - epochs: e.g., 2–5
  - batch_size: 8–32 (GPU memory dependent)
  - learning_rate: e.g., 2e-5
  - optimizer: AdamW
  - scheduler: linear with warmup

Adjust hyperparameters depending on dataset size and GPU availability. The notebook includes training loops and evaluation steps.

## Outputs
The notebook produces:
- Training/validation loss and accuracy plots
- Classification report (precision, recall, F1)
- Confusion matrix visualization
- Saved model checkpoint and tokenizer (paths shown in the notebook)

Check the notebook's final cells to see where the model and tokenizer are saved and how to load them for inference.

## Reproduce results
To reproduce:
1. Ensure the same random seed is set in the notebook.
2. Use the same dataset split and preprocessing.
3. Use matching library versions (PyTorch, transformers). Note that different versions can change training dynamics.
4. Run all cells in order, ideally on a GPU.

If you want, I can extract exact library versions used in the notebook and add a requirements.txt.

## File structure
- Spam_Text_classication_roberta_base_model.ipynb — main notebook with the full pipeline
- README.md — this file (replace)

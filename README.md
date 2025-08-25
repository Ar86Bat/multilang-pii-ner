
# Multilingual PII NER

This project fine-tunes a multilingual transformer model (`xlm-roberta-base`) for Named Entity Recognition (NER) to detect and mask Personally Identifiable Information (PII) in text across multiple languages (English, German, Italian, French).
The project is finalized and ready for deployment or sharing on the Hugging Face Hub.


## Project Structure

- `notebooks/`
   - `01_eda_openpii.ipynb`: Exploratory Data Analysis (EDA) of the PII dataset, including data overview, label distribution, and sample inspection.
   - `02_preprocessing_conll.ipynb`: Preprocesses and tokenizes the dataset in CoNLL format, aligns labels, and saves ready-to-train data.
   - `03_training.ipynb`: Fine-tunes the model on the preprocessed data and evaluates performance.
   - `04_validation.ipynb`: Validates the trained model and analyzes results.
- `data/`
   - `train.conll`: Training data in CoNLL format.
   - `validation.conll`: Validation data in CoNLL format.
- `model/`
   - `config.json`, `model.safetensors`, `sentencepiece.bpe.model`, `special_tokens_map.json`, `tokenizer_config.json`, `tokenizer.json`, `training_args.bin`, `validation_results`: All model artifacts and outputs.
- `requirements.txt`: Python dependencies.
- `Dockerfile`: For containerized reproducible environments.
- `.gitignore`, `.dockerignore`: Ignore rules for git and Docker.


## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ar86Bat/multilang-pii-ner.git
   cd multilang-pii-ner
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```


3. **(Optional) Use Docker**
   ```bash
   docker build -t multilang-pii-ner .
   docker run -it --rm multilang-pii-ner
   ```

## Usage

### 1. Exploratory Data Analysis (EDA)

Run `01_eda_openpii.ipynb` to:
- Explore the structure and contents of the PII dataset.
- Visualize label distributions and inspect sample records.
- Understand the data before preprocessing and modeling.


### 2. Preprocessing

Run `02_preprocessing_conll.ipynb` to:
- Load and filter the dataset for target languages.
- Detect and normalize entity spans.
- Build BIO tag mappings.
- Tokenize and align labels.
- Save processed data for training in CoNLL format.


### 3. Training

Run `03_training.ipynb` to:
- Load the tokenized dataset and label mappings.
- Initialize and fine-tune the `xlm-roberta-base` model for token classification.
- Evaluate model performance using standard NER metrics.
- Save the trained model and tokenizer to the `model/` directory.

### 4. Validation

Run `04_validation.ipynb` to:
- Validate the trained model on the validation set.
- Analyze and visualize the results.


## Data

- Uses the [ai4privacy/open-pii-masking-500k-ai4privacy](https://huggingface.co/datasets/ai4privacy/open-pii-masking-500k-ai4privacy) dataset.
- Preprocessing and training scripts expect data in the `data/` directory in CoNLL format.



## Results & Model Artifacts

- The model is evaluated using the `seqeval` metric (precision, recall, F1) on validation data.
- All model artifacts (trained model, tokenizer, config, etc.) are saved in the `model/` directory.

### Validation Results Analysis

The model demonstrates strong performance across most PII entity types. Below is a summary of the validation results (see `model/validation_results` for full details):

- **Overall accuracy:** 99.24%
- **Macro F1-score:** 0.954
- **Weighted F1-score:** 0.992

**Entity-level highlights:**

- High F1-scores (>0.97) for common entities such as `AGE`, `BUILDINGNUM`, `CITY`, `DATE`, `EMAIL`, `GIVENNAME`, `STREET`, `TELEPHONENUM`, and `TIME`.
- Excellent performance on `EMAIL` and `DATE` (F1 ≈ 0.999).
- Slightly lower F1-scores for more challenging or less frequent entities:
   - `DRIVERLICENSENUM` (F1 ≈ 0.85)
   - `GENDER` (F1 ≈ 0.83)
   - `PASSPORTNUM` (F1 ≈ 0.88)
   - `SURNAME` (F1 ≈ 0.85)
   - `SEX` (F1 ≈ 0.84)
- The model handles both B- and I- tags well, with some drop in performance for rare I- tags due to limited support.

**Conclusion:**

The model is highly effective for multilingual PII NER, especially for frequent and well-represented entity types. For rare or ambiguous entities, further data augmentation or targeted fine-tuning may improve results.

## Hugging Face Hub

The trained model from this project is available at: [https://huggingface.co/Ar86Bat/multilang-pii-ner](https://huggingface.co/Ar86Bat/multilang-pii-ner)

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.

## License

MIT License
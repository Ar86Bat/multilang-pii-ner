# Multilingual PII NER

This project fine-tunes a multilingual transformer model (`xlm-roberta-base`) for Named Entity Recognition (NER) to detect and mask Personally Identifiable Information (PII) in text across multiple languages (English, German, Italian, French).

## Project Structure

- `notebooks/`
  - `01_eda_openpii.ipynb`: Exploratory Data Analysis (EDA) of the PII dataset, including data overview, label distribution, and sample inspection.
  - `02_preprocessing.ipynb`: Preprocesses and tokenizes the dataset, aligns labels, and saves ready-to-train data.
  - `03_training.ipynb`: Fine-tunes the model on the preprocessed data and evaluates performance.
- `data/`: Contains datasets, tokenized data, and label mappings.
- `requirements.txt`: Python dependencies.
- `Dockerfile`: For containerized reproducible environments.

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

Run `02_preprocessing.ipynb` to:
- Load and filter the dataset for target languages.
- Detect and normalize entity spans.
- Build BIO tag mappings.
- Tokenize and align labels.
- Save processed data for training.

### 3. Training

Run `03_training.ipynb` to:
- Load the tokenized dataset and label mappings.
- Initialize and fine-tune the `xlm-roberta-base` model for token classification.
- Evaluate model performance using standard NER metrics.

## Data

- Uses the [ai4privacy/open-pii-masking-500k-ai4privacy](https://huggingface.co/datasets/ai4privacy/open-pii-masking-500k-ai4privacy) dataset.
- Preprocessing and training scripts expect data in the `data/` directory.

## Results

- The model is evaluated using the `seqeval` metric (precision, recall, F1) on validation data.
- Artifacts (trained models, logs) are saved in the project directory.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.

## License

MIT License
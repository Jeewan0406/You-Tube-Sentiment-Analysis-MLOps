import os
import re
import logging
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# ─── LOGGING SETUP ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('preprocessing.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ─── NLTK RESOURCES ─────────────────────────────────────────────────────────────
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# ─── PREPROCESSING FUNCTION ─────────────────────────────────────────────────────
def preprocess_comment(comment):
    try:
        if pd.isna(comment):
            return ""
        comment = str(comment).strip()
        if not comment:
            return ""

        comment = comment.lower()
        comment = re.sub(r'\s+', ' ', comment)                  # normalize whitespace
        comment = re.sub(r'[^a-z0-9\s!?.,]', '', comment)       # keep only useful chars

        tokens = comment.split()

        # Preserve important negation words
        stop_words = set(stopwords.words('english')) - {'not', 'no', 'nor', 'but', 'however', 'yet', 'although'}
        tokens = [word for word in tokens if word not in stop_words]

        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        cleaned = ' '.join(tokens).strip()
        return cleaned if cleaned else ""

    except Exception as e:
        logger.error(f"Error preprocessing comment: {e}")
        return ""

# ─── AUTO-DETECT TEXT COLUMN & NORMALIZE ────────────────────────────────────────
def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if df.shape[1] != 2:
            raise ValueError(f"Expected exactly 2 columns, found {df.shape[1]}")

        cols = df.columns.tolist()
        logger.debug(f"Columns detected: {cols}")

        # Auto-detect which column contains the text (the one that is NOT numeric-only and longer)
        text_col = None
        label_col = None

        for col in cols:
            sample = df[col].astype(str).str.len()
            if sample.mean() > 20:        # comments are usually long
                text_col = col
            else:
                label_col = col

        # Fallback if above logic fails (very rare)
        if text_col is None:
            text_col = cols[0]   # assume first column is text
            label_col = cols[1]

        logger.debug(f"Automatically selected text column: '{text_col}'")
        logger.debug(f"Label column: '{label_col}'")

        df[text_col] = df[text_col].apply(preprocess_comment)

        logger.debug("Text preprocessing completed successfully")
        return df

    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise

# ─── SAVE FUNCTION ──────────────────────────────────────────────────────────────
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        interim_path = os.path.join(data_path, 'interim')
        os.makedirs(interim_path, exist_ok=True)

        train_path = os.path.join(interim_path, 'train_processed.csv')
        test_path  = os.path.join(interim_path, 'test_processed.csv')

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        logger.debug(f"Successfully saved:\n  {train_path}\n  {test_path}")

    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

# ─── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    try:
        logger.debug("Starting preprocessing pipeline")

        raw_dir = 'data/raw'
        train_path = os.path.join(raw_dir, 'train.csv')
        test_path  = os.path.join(raw_dir, 'test.csv')

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Missing {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Missing {test_path}")

        logger.debug("Loading raw data...")
        train_data = pd.read_csv(train_path)
        test_data  = pd.read_csv(test_path)

        logger.debug(f"Loaded → train: {train_data.shape}, test: {test_data.shape}")

        logger.debug("Normalizing text (auto-detecting column)...")
        train_processed = normalize_text(train_data.copy())
        test_processed  = normalize_text(test_data.copy())

        logger.debug("Saving processed data...")
        save_data(train_processed, test_processed, data_path='data')

        logger.debug("Preprocessing pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
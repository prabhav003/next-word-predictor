import re
import random
from pathlib import Path


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_FILE = DATA_DIR / "next_word_predictor.txt"

SEED = 42

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10

MIN_PARAGRAPH_LENGTH = 20


# --------------------------------------------------
# 2. CLEAN TEXT
# --------------------------------------------------

def clean_text(text):
    """Remove known dataset-generation instructions."""

    patterns = [
        r"(?im)^.*(?:generated upon confirmation|sample of the \d[\d,]*-word dataset|complete dataset is now ready for download).*$",
        r"(?im)^.*please let me know if you have any other requests.*$",
        r"(?im)^.*your RNN model can use this.*$",
    ]

    for pattern in patterns:
        text = re.sub(pattern, "", text)

    return text.strip()


# --------------------------------------------------
# 3. REMOVE EXACT DUPLICATES
# --------------------------------------------------

def remove_duplicates(paragraphs):
    """
    Remove short paragraphs and exact duplicates.

    Uses a set for fast duplicate detection.
    """

    unique = []
    seen = set()

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if len(paragraph) < MIN_PARAGRAPH_LENGTH:
            continue

        normalized = paragraph.lower()

        if normalized in seen:
            continue

        seen.add(normalized)
        unique.append(paragraph)

    return unique


# --------------------------------------------------
# 4. PREPARE DATASET
# --------------------------------------------------

def main():

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    print("Reading dataset...")

    text = INPUT_FILE.read_text(encoding="utf-8")

    original_characters = len(text)

    print(f"Original characters: {original_characters:,}")

    print("Cleaning text...")

    text = clean_text(text)

    print("Extracting paragraphs...")

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    original_count = len(paragraphs)

    print(f"Original paragraphs: {original_count:,}")

    print("Removing exact duplicates...")

    paragraphs = remove_duplicates(paragraphs)

    unique_count = len(paragraphs)

    print(f"Unique paragraphs: {unique_count:,}")

    if unique_count < 3:
        raise ValueError(
            f"Only {unique_count} unique paragraphs found."
        )

    # --------------------------------------------------
    # 5. SHUFFLE
    # --------------------------------------------------

    print("Shuffling paragraphs...")

    random.Random(SEED).shuffle(paragraphs)

    # --------------------------------------------------
    # 6. SPLIT DATASET
    # --------------------------------------------------

    train_end = int(unique_count * TRAIN_RATIO)

    validation_end = int(
        unique_count * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    splits = {
        "train.txt": paragraphs[:train_end],
        "validation.txt": paragraphs[train_end:validation_end],
        "test.txt": paragraphs[validation_end:],
    }

    # --------------------------------------------------
    # 7. SAVE DATASETS
    # --------------------------------------------------

    print("Saving datasets...")

    for filename, split_data in splits.items():

        output_file = DATA_DIR / filename

        output_file.write_text(
            "\n\n".join(split_data),
            encoding="utf-8"
        )

        print(
            f"{filename}: "
            f"{len(split_data):,} paragraphs"
        )

    # --------------------------------------------------
    # 8. SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 40)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 40)

    print(f"Original paragraphs: {original_count:,}")
    print(f"Unique paragraphs:   {unique_count:,}")

    for filename, split_data in splits.items():
        print(f"{filename}: {len(split_data):,}")

    print("=" * 40)


if __name__ == "__main__":
    main()
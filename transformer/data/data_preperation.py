

'''

Dataset Preparation

1. Import libraries: Import re for text cleaning, random for shuffling, Path for file paths, and SequenceMatcher for detecting similar paragraphs.
2. Define file paths: Locate next_word_predictor.txt in the same directory as the Python script.
3. Set random seed: Use SEED = 42 to ensure reproducible shuffling and consistent dataset splits.
4. Clean the text: Remove irrelevant dataset-generation notes, sample announcements, and unnecessary instructions using regular expressions.
5. Extract paragraphs: Split the cleaned text into individual paragraphs using blank lines and remove empty paragraphs.
6. Remove duplicates: Eliminate exact duplicate paragraphs and near-duplicates with a similarity ratio greater than 90%. Also, discard paragraphs shorter than 20 characters.
7. Validate the dataset: Ensure at least three unique paragraphs remain before splitting.
8. Shuffle the paragraphs: Randomly shuffle the paragraphs using the fixed seed to avoid preserving their original order.
9. Split the dataset: Divide the paragraphs into three sets:
    * Training set: 80% of the data.
    * Validation set: 10% of the data.
    * Test set: Remaining 10% of the data.
10. Save the datasets: Write the three splits to train.txt, validation.txt, and test.txt in the data directory.
11. Display statistics: Print the original paragraph count, number of unique paragraphs, and number of paragraphs in each split.
12. Execute the pipeline: Run main() when the script is executed directly.

'''


import re
import random
from pathlib import Path
from difflib import SequenceMatcher

DATA_DIR = Path(__file__).parent
INPUT_FILE = DATA_DIR / "next_word_predictor.txt"

SEED = 42


def clean_text(text):
    # Remove dataset-generation notes
    patterns = [
        r"(?im)^.*(?:generated upon confirmation|sample of the \d[\d,]*-word dataset|complete dataset is now ready for download).*$",
        r"(?im)^.*please let me know if you have any other requests.*$",
        r"(?im)^.*your RNN model can use this.*$",
    ]

    for pattern in patterns:
        text = re.sub(pattern, "", text)

    return text.strip()


def remove_duplicates(paragraphs):
    unique = []
    seen = set()

    for paragraph in paragraphs:
        normalized = paragraph.lower().strip()

        if len(normalized) < 20 or normalized in seen:
            continue

        if any(
            SequenceMatcher(None, normalized, existing).ratio() > 0.90
            for existing in unique
        ):
            continue

        seen.add(normalized)
        unique.append(paragraph.strip())

    return unique


def main():
    text = INPUT_FILE.read_text(encoding="utf-8")
    text = clean_text(text)

    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    original_count = len(paragraphs)
    paragraphs = remove_duplicates(paragraphs)

    if len(paragraphs) < 3:
        raise ValueError(
            f"Only {len(paragraphs)} unique paragraphs found."
        )

    random.Random(SEED).shuffle(paragraphs)

    train_end = int(len(paragraphs) * 0.80)
    val_end = int(len(paragraphs) * 0.90)

    splits = {
        "train.txt": paragraphs[:train_end],
        "validation.txt": paragraphs[train_end:val_end],
        "test.txt": paragraphs[val_end:],
    }

    for filename, data in splits.items():
        (DATA_DIR / filename).write_text(
            "\n\n".join(data), encoding="utf-8"
        )

    print(f"Original paragraphs: {original_count}")
    print(f"Unique paragraphs: {len(paragraphs)}")

    for filename, data in splits.items():
        print(f"{filename}: {len(data)} paragraphs")

    print("Dataset preparation complete!")


if __name__ == "__main__":
    main()
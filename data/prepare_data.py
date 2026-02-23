from datasets import load_dataset
import os

HF_DATASET_NAME = os.getenv("HF_DATASET_NAME", os.getenv("DATASET_NAME", "")).strip()
HF_DATASET_CONFIG = os.getenv("HF_DATASET_CONFIG", os.getenv("DATASET_CONFIG", "")).strip() or None
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}


def _resolve_column(columns: list[str], candidates: list[str], kind: str) -> str:
    available_lower = {name.lower(): name for name in columns}
    for candidate in candidates:
        found = available_lower.get(candidate.lower())
        if found:
            return found
    raise ValueError(f"Unable to find {kind} column in dataset. Available columns: {columns}")


def _to_label_id(value: str | int) -> int:
    if isinstance(value, int):
        return value
    normalized = str(value).strip().lower()
    if normalized in LABEL2ID:
        return LABEL2ID[normalized]
    raise ValueError(f"Unsupported sentiment label value: {value}")


def main():
    os.makedirs("data", exist_ok=True)
    if not HF_DATASET_NAME:
        raise ValueError(
            "HF dataset name is required for prepare_data.py. "
            "Set HF_DATASET_NAME (or DATASET_NAME)."
        )
    dataset = load_dataset(HF_DATASET_NAME, HF_DATASET_CONFIG)

    if "train" not in dataset:
        raise ValueError("Dataset must contain a 'train' split")
    if "test" not in dataset:
        split = dataset["train"].train_test_split(test_size=TEST_SIZE, seed=RANDOM_SEED)
        dataset = {"train": split["train"], "test": split["test"]}
    else:
        dataset = {"train": dataset["train"], "test": dataset["test"]}

    columns = list(dataset["train"].column_names)
    text_col = _resolve_column(columns, ["sentence", "text", "headline"], "text")
    sentiment_col = _resolve_column(columns, ["label", "sentiment"], "sentiment")

    def normalize(batch):
        return {
            "sentence": batch[text_col],
            "label": [_to_label_id(item) for item in batch[sentiment_col]],
        }

    train = dataset["train"].map(normalize, batched=True, remove_columns=dataset["train"].column_names)
    test = dataset["test"].map(normalize, batched=True, remove_columns=dataset["test"].column_names)

    train.to_json("data/train.json")
    test.to_json("data/test.json")

    print("Data prepared successfully!")
    print(f"Source dataset: {HF_DATASET_NAME}")
    print("Saved files:")
    print(" - data/train.json")
    print(" - data/test.json")
    print(f"Train samples: {len(train)}")
    print(f"Test samples: {len(test)}")

if __name__ == "__main__":
    main()

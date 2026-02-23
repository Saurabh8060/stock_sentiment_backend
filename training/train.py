import os
from collections.abc import Iterable

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler

MODEL_NAME = os.getenv("MODEL_NAME", "ProsusAI/finbert")
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
HF_DATASET_NAME = os.getenv("HF_DATASET_NAME", os.getenv("DATASET_NAME", "")).strip()
HF_DATASET_CONFIG = os.getenv("HF_DATASET_CONFIG", os.getenv("DATASET_CONFIG", "")).strip() or None
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "3"))
TRAIN_BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "16"))
EVAL_BATCH_SIZE = int(os.getenv("EVAL_BATCH_SIZE", "16"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-5"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.01"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))
DATA_CACHE_DIR = os.getenv("HF_DATASETS_CACHE", os.path.join(os.getcwd(), ".hf_datasets_cache"))

def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute accuracy, precision, recall, and F1 score."""
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _evaluate(
    model: AutoModelForSequenceClassification,
    dataloader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            moved = _to_device(batch, device)
            labels = moved.pop("labels")
            outputs = model(**moved)
            logits = outputs.logits.detach().cpu().numpy()
            all_logits.append(logits)
            all_labels.append(labels.detach().cpu().numpy())

    if not all_logits:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    stacked_logits = np.concatenate(all_logits, axis=0)
    stacked_labels = np.concatenate(all_labels, axis=0)
    return compute_metrics(stacked_logits, stacked_labels)


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_column(columns: list[str], candidates: list[str], kind: str) -> str:
    available_lower = {name.lower(): name for name in columns}
    for candidate in candidates:
        found = available_lower.get(candidate.lower())
        if found:
            return found
    raise ValueError(f"Unable to find {kind} column in dataset. Available columns: {columns}")


def _to_label_id(value: str | int) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    normalized = str(value).strip().lower()
    if normalized in LABEL2ID:
        return LABEL2ID[normalized]
    raise ValueError(f"Unsupported sentiment label value: {value}")


def _load_hf_dataset():
    if not HF_DATASET_NAME:
        raise ValueError(
            "HF dataset name is empty. Set HF_DATASET_NAME (or DATASET_NAME)."
        )
    dataset = load_dataset(HF_DATASET_NAME, HF_DATASET_CONFIG, cache_dir=DATA_CACHE_DIR)
    return dataset, f"huggingface:{HF_DATASET_NAME}"


def _load_training_dataset():
    dataset, source = _load_hf_dataset()

    if "train" not in dataset:
        raise ValueError("Dataset must contain a 'train' split.")
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
            "labels": [_to_label_id(item) for item in batch[sentiment_col]],
        }

    normalized = {
        "train": dataset["train"].map(normalize, batched=True),
        "test": dataset["test"].map(normalize, batched=True),
    }

    keep_cols = {"sentence", "labels"}
    for split_name in ("train", "test"):
        remove_cols = [c for c in normalized[split_name].column_names if c not in keep_cols]
        if remove_cols:
            normalized[split_name] = normalized[split_name].remove_columns(remove_cols)

    return normalized, source


def main() -> None:
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = DATA_CACHE_DIR

    # 1) Load dataset (HF dataset first, local JSON fallback)
    dataset, dataset_source = _load_training_dataset()

    # 2) Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 3) Tokenization function
    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )

    # 4) Tokenize dataset
    train_ds = dataset["train"].map(tokenize, batched=True)
    test_ds = dataset["test"].map(tokenize, batched=True)

    # 6) Set PyTorch format
    train_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )
    test_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    # 7) Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )
    device = _resolve_device()
    model.to(device)

    train_loader = DataLoader(train_ds, shuffle=True, batch_size=TRAIN_BATCH_SIZE)
    eval_loader = DataLoader(test_ds, batch_size=EVAL_BATCH_SIZE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    total_steps = max(1, NUM_EPOCHS * len(train_loader))
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    print(f"Dataset source: {dataset_source}")
    print(f"Training on device: {device}")
    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    # 8) Train
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            moved = _to_device(batch, device)
            outputs = model(**moved)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            if step % 25 == 0 or step == len(train_loader):
                print(f"Epoch {epoch + 1}/{NUM_EPOCHS} step {step}/{len(train_loader)} loss={loss.item():.4f}")

        avg_loss = running_loss / max(1, len(train_loader))
        metrics = _evaluate(model, eval_loader, device)
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} - "
            f"loss: {avg_loss:.4f}, "
            f"acc: {metrics['accuracy']:.4f}, "
            f"precision: {metrics['precision']:.4f}, "
            f"recall: {metrics['recall']:.4f}, "
            f"f1: {metrics['f1']:.4f}"
        )

    # 9) Save model locally
    model.save_pretrained("model")
    tokenizer.save_pretrained("model")

    print("Training complete")
    print("Model saved to ./model")

if __name__ == "__main__":
    main()

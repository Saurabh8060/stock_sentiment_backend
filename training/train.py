from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

MODEL_NAME = "roberta-base"

def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, and F1 score"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    # 1️⃣ Load prepared JSON data
    dataset = load_dataset(
        "json",
        data_files={
            "train": "data/train.json",
            "test": "data/test.json"
        }
    )

    # 2️⃣ Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 3️⃣ Tokenization function
    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    # 4️⃣ Tokenize dataset
    dataset = dataset.map(tokenize, batched=True)

    # 5️⃣ VERY IMPORTANT: rename label → labels
    dataset = dataset.rename_column("label", "labels")

    # 6️⃣ Set PyTorch format
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    # 7️⃣ Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    )

    # 8️⃣ Training arguments (transformers 5.0 compatible)
    args = TrainingArguments(
        output_dir="outputs",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="logs",
        report_to="none"
    )

    # 9️⃣ Trainer (transformers 5.0: use processing_class instead of tokenizer)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,  # ← Changed for transformers 5.0
        compute_metrics=compute_metrics
    )

    # 🔟 Train
    trainer.train()

    # 1️⃣1️⃣ Save model locally
    trainer.save_model("model")
    tokenizer.save_pretrained("model")

    print("✅ Training complete")
    print("✅ Model saved to ./model")

if __name__ == "__main__":
    main()
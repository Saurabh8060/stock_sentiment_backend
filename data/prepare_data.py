from datasets import load_dataset
import os

def main():
    os.makedirs("data", exist_ok=True)
    dataset = load_dataset("atrost/financial_phrasebank")
    
    dataset["train"].to_json("data/train.json")
    dataset["test"].to_json("data/test.json")

    print("Data prepared successfully!")
    print("Saved files:")
    print(" - data/train.json")
    print(" - data/test.json")
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")
    print(f"Validation samples: {len(dataset['validation'])}")

if __name__ == "__main__":
    main()
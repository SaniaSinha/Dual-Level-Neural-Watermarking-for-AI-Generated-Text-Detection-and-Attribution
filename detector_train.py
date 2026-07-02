import random
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import DETECTOR_MODEL_NAME, DETECTOR_SAVE_PATH, MAX_LENGTH, SEED
from data_loader import load_balanced_dataset, split_dataset


class AIDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index):
        item = {
            key: torch.tensor(value[index])
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(prediction):
    logits = prediction.predictions
    labels = prediction.label_ids
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds)
    }


def train_detector():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    df = load_balanced_dataset()
    train_texts, test_texts, train_labels, test_labels = split_dataset(df)

    tokenizer = AutoTokenizer.from_pretrained(DETECTOR_MODEL_NAME)

    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    train_dataset = AIDetectionDataset(train_encodings, train_labels)
    test_dataset = AIDetectionDataset(test_encodings, test_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        DETECTOR_MODEL_NAME,
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir="/kaggle/working/detector_training_results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        seed=SEED
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate()

    print("Final detector metrics:")
    print(metrics)

    model.save_pretrained(DETECTOR_SAVE_PATH)
    tokenizer.save_pretrained(DETECTOR_SAVE_PATH)

    print("Detector saved to:", DETECTOR_SAVE_PATH)


if __name__ == "__main__":
    train_detector()

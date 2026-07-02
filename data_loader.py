import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from config import DATASET_NAME, TRAIN_SAMPLES_PER_CLASS, TEST_SIZE, SEED


def load_balanced_dataset():
    dataset = load_dataset(DATASET_NAME)
    df = pd.DataFrame(dataset["train"])

    df["label"] = df["source"].map({
        "human": 0,
        "ai": 1
    })

    df = df.dropna(subset=["text", "label"])

    human_df = df[df["label"] == 0].sample(
        TRAIN_SAMPLES_PER_CLASS,
        random_state=SEED
    )

    ai_df = df[df["label"] == 1].sample(
        TRAIN_SAMPLES_PER_CLASS,
        random_state=SEED
    )

    df_small = pd.concat([human_df, ai_df])
    df_small = df_small.sample(frac=1, random_state=SEED).reset_index(drop=True)

    return df_small


def split_dataset(df):
    return train_test_split(
        df["text"].tolist(),
        df["label"].astype(int).tolist(),
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["label"]
    )

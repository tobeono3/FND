### fake-news-detector/src/train.py

import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load and merge data
df_fake = pd.read_csv("data/Fake.csv")
df_real = pd.read_csv("data/True.csv")
df_fake["label"] = 0
df_real["label"] = 1
data = pd.concat([df_fake, df_real]).sample(frac=1).reset_index(drop=True)

# Use only the 'text' column
texts = data['text'].tolist()
labels = data['label'].tolist()

# Create Hugging Face Dataset
dataset = Dataset.from_dict({"text": texts, "label": labels})

# Tokenization
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.train_test_split(test_size=0.2)

# Load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Compute metrics for evaluation
def compute_metrics(p):
    preds = p.predictions.argmax(axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Training arguments
training_args = TrainingArguments(
    output_dir="models/results",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    compute_metrics=compute_metrics
)

trainer.train()

# Evaluate and print final metrics
metrics = trainer.evaluate()
print(metrics)

# Save model
model.save_pretrained("models/fake-news-bert")
tokenizer.save_pretrained("models/fake-news-bert")

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json

# ----------------------
# Step 1: Load the Dataset
# ----------------------
print("Loading the dataset...")
dataset = load_dataset("MidhunKanadan/logical-fallacy-classification")

# ----------------------
# Step 2: Define Label Mapping
# ----------------------
label2id = {
    "equivocation": 0,
    "faulty generalization": 1,
    "fallacy of logic": 2,
    "ad populum": 3,
    "circular reasoning": 4,
    "false dilemma": 5,
    "false causality": 6,
    "fallacy of extension": 7,
    "fallacy of credibility": 8,
    "fallacy of relevance": 9,
    "intentional": 10,
    "appeal to emotion": 11,
    "ad hominem": 12,
}
id2label = {v: k for k, v in label2id.items()}

# Map string labels to integers
def map_labels(example):
    example["label"] = label2id[example["label"].lower()]
    return example

print("Mapping labels to integers...")
dataset = dataset.map(map_labels)

# ----------------------
# Step 3: Load Tokenizer
# ----------------------
print("Loading the tokenizer...")
model_name = "roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ----------------------
# Step 4: Tokenize the Dataset
# ----------------------
print("Tokenizing the dataset...")
def tokenize_function(example):
    return tokenizer(example["statement"], padding=True, truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# ----------------------
# Step 5: Compute Class Weights for Imbalanced Data
# ----------------------
print("Computing class weights...")
class_weights = torch.tensor(
    [
        len(dataset["train"]) / (len(dataset["train"].filter(lambda x: x["label"] == i)) + 1)
        for i in range(len(label2id))
    ],
    dtype=torch.float
).cuda()

# ----------------------
# Step 6: Define Custom Model with Weighted Loss
# ----------------------
print("Defining the custom model...")
class CustomRobertaModel(torch.nn.Module):
    def __init__(self, model_name, num_labels, class_weights):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.class_weights = class_weights

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        return {"loss": loss, "logits": logits}

model = CustomRobertaModel(model_name=model_name, num_labels=len(label2id), class_weights=class_weights)

# ----------------------
# Step 7: Define Metrics for Evaluation
# ----------------------
print("Defining evaluation metrics...")
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=1)
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

# ----------------------
# Step 8: Define Training Arguments
# ----------------------
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-6,
    lr_scheduler_type="linear",
    warmup_steps=500,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=15,
    weight_decay=0.01,
    fp16=True,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=1,
)

# ----------------------
# Step 9: Initialize Trainer
# ----------------------
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["dev"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ----------------------
# Step 10: Train the Model
# ----------------------
print("Starting training...")
trainer.train()

# ----------------------
# Step 11: Save the Model, Tokenizer, and Label Mapping
# ----------------------
print("Saving the model and tokenizer...")
model.model.save_pretrained("./results")
tokenizer.save_pretrained("./results")

# Save label mapping for easy use
with open("./results/label_mapping.json", "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f)

print("Training complete! Model, tokenizer, and label mappings saved in './results'")
from datasets import load_metric, Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification
)
from sklearn.metrics import f1_score
import torch as nn
import numpy as np
import pandas as pd
import json

model_checkpoint = "allenai/scibert_scivocab_uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_function_batch(examples):
    return tokenizer(
        examples["sentence"], 
        truncation=True,
        padding=True,
        max_length=512,
        #add_special_tokens=True,
        return_tensors="pt"
    )

###data

label2id = {"BACKGROUND": 0, "OBJECTIVE": 1, "METHODS": 2, "RESULTS": 3, "CONCLUSIONS": 4}

with open('data/train.txt') as f:
    train_lines = f.readlines()    
train_data = []
for line in train_lines:
    new_line = {
        "sentence": line.split("\t")[2][:-2], 
        "label": label2id[line.split("\t")[1]]
    }
    train_data.append(new_line)

with open('data/validation.txt') as f:
    eval_lines = f.readlines()    
eval_data = []
for line in eval_lines:
    new_line = {
        "sentence": line.split("\t")[2][:-2], 
        "label": label2id[line.split("\t")[1]]
    }
    eval_data.append(new_line)

with open('data/test.txt') as f:
    test_lines = f.readlines()    
test_data = []
for line in test_lines:
    new_line = {
        "sentence": line.split("\t")[2][:-2], 
        "label": label2id[line.split("\t")[1]]
    }
    test_data.append(new_line)

train_dataset = Dataset.from_list(train_data)
eval_dataset = Dataset.from_list(eval_data)
test_dataset = Dataset.from_list(test_data)

train_encoded = train_dataset.map(preprocess_function_batch, batched=True)
eval_encoded = eval_dataset.map(preprocess_function_batch, batched=True)
test_encoded = test_dataset.map(preprocess_function_batch, batched=True)

final_train = train_encoded.rename_column("label", "labels")
final_eval = eval_encoded.rename_column("label", "labels")
final_test = test_encoded.rename_column("label", "labels")

train_df = final_train.to_pandas()
class_weights = (1 - (train_df["labels"].value_counts().sort_index() / len(train_df))).values
class_weights

class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        loss_func = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_func(logits, labels)

        return (loss, outputs) if return_outputs else loss
    
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    return {"f1": f1}


def train_test_hyperparams(dropout, dir, wd, lr, ws, bs):
    print("##########")

    print(f"Training {dir}: drop: {dropout} | weight_dec: {wd} | lr: {lr} | warmup: {ws} | batchsize: {bs}")
    
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=5, hidden_dropout_prob=dropout)
    model.to("cuda")
    model_output_dir = f"scibert-finetuned-abstract-classification-hyperparam-{dir}"

    args = TrainingArguments(
        output_dir=model_output_dir,
        evaluation_strategy="steps",
        eval_steps=50,
        logging_strategy="steps",
        logging_steps=50,
        #save_strategy="steps",
        #save_steps=100,
        learning_rate=lr,
        weight_decay=wd,
        warmup_steps=ws,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        num_train_epochs=6,
        load_best_model_at_end=True,
        # fp16=True, 
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=final_train,
        eval_dataset=final_eval,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # train the model & save checkpoint

    trainer.train()
    model.save_pretrained(model_output_dir + "/best_model")

    #evaluate on test
    metric = load_metric("accuracy")

    dataset_test_encoded = final_test
    test_predictions = trainer.predict(dataset_test_encoded)
    test_predictions_argmax = np.argmax(test_predictions[0], axis=1)
    test_references = np.array(final_test["labels"])
    # Compute accuracy & f1
    acc = metric.compute(predictions=test_predictions_argmax, references=test_references)["accuracy"]
    f1 = f1_score(test_references, test_predictions_argmax, average="weighted")
    print("Test Results:")
    print("accuracy:", acc)
    print("f1-score:", f1)
    output_json = {
        "dir": dir,
        "batchsize": bs,
        "dropout": dropout,
        "warmup": ws,
        "weight decay": wd,
        "learningrate": lr,
        "acc": acc,
        "f1": f1
    }
    with open(f"{model_output_dir}/metrics.json", "w") as outfile:
        json.dump(output_json, outfile)


if __name__ == "__main__":
    lrs = [2e-5, 3e-5, 4e-5]
    wds = [0, 0.08, 0.18]
    wss = [0, 100]
    bs = [16, 32, 64]
    dropouts = [0.1, 0.25]

    for i, lr in enumerate(lrs):
        for j, wd in enumerate(wds):
            for k, ws in enumerate(wss):
                for g, b in enumerate(bs):
                    for e, do in enumerate(dropouts):
                        dir_no = f"{i}-{j}-{k}-{g}-{e}"
                        train_test_hyperparams(do, dir_no, wd, lr, ws, b)


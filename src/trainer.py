import torch
import numpy as np
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

def compute_metrics(eval_pred):
    """Calcule les métriques"""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"f1": f1}

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

def create_trainer(model, hf, tokenizer, class_weights):
    """Crée et configure le trainer"""
    
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=True,
        logging_steps=20,
        report_to="none",
        save_total_limit=1,
        seed=42,
        warmup_steps=20,
        lr_scheduler_type="linear",
    )
    
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=hf["train"],
        eval_dataset=hf["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )
    trainer.class_weights = class_weights
    
    return trainer
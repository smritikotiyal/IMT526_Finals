import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
import gc
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create results directory if it doesn't exist
RESULTS_DIR = './training_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Custom dataset class
class ComplaintDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.training_loss = []
        self.validation_metrics = []
        self.learning_rates = []
        self.current_step = 0
    
    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if logs is not None:
            if 'loss' in logs:
                self.training_loss.append((self.current_step, logs['loss']))
            if 'learning_rate' in logs:
                self.learning_rates.append((self.current_step, logs['learning_rate']))
            if 'eval_loss' in logs:
                self.validation_metrics.append({
                    'step': self.current_step,
                    'eval_loss': logs['eval_loss'],
                    'eval_accuracy': logs.get('eval_accuracy', None),
                    'eval_f1': logs.get('eval_f1', None),
                    'eval_precision': logs.get('eval_precision', None),
                    'eval_recall': logs.get('eval_recall', None)
                })
            self.current_step += 1
        return control

def plot_training_curves(metrics_callback, save_dir):
    # Plot training loss
    plt.figure(figsize=(10, 6))
    steps, losses = zip(*metrics_callback.training_loss)
    plt.plot(steps, losses, label='Training Loss')
    plt.title('Training Loss Over Time')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{save_dir}/training_loss.png')
    plt.close()

    # Plot learning rate
    plt.figure(figsize=(10, 6))
    steps, lrs = zip(*metrics_callback.learning_rates)
    plt.plot(steps, lrs, label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.savefig(f'{save_dir}/learning_rate.png')
    plt.close()

    # Plot validation metrics
    if metrics_callback.validation_metrics:
        metrics_df = pd.DataFrame(metrics_callback.validation_metrics)
        plt.figure(figsize=(12, 6))
        for column in ['eval_accuracy', 'eval_f1', 'eval_precision', 'eval_recall']:
            if column in metrics_df.columns:
                plt.plot(metrics_df['step'], metrics_df[column], label=column.replace('eval_', ''))
        plt.title('Validation Metrics Over Time')
        plt.xlabel('Step')
        plt.ylabel('Score')
        plt.legend()
        plt.savefig(f'{save_dir}/validation_metrics.png')
        plt.close()

def plot_confusion_matrix(y_true, y_pred, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{save_dir}/confusion_matrix.png')
    plt.close()

def plot_roc_curve(y_true, y_prob, save_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'{save_dir}/roc_curve.png')
    plt.close()

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = torch.softmax(torch.tensor(pred.predictions), dim=1)[:, 1].numpy()
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    
    # Store predictions and probabilities for later visualization
    if not hasattr(compute_metrics, "all_labels"):
        compute_metrics.all_labels = []
        compute_metrics.all_preds = []
        compute_metrics.all_probs = []
    
    compute_metrics.all_labels.extend(labels)
    compute_metrics.all_preds.extend(preds)
    compute_metrics.all_probs.extend(probs)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

try:
    # Create a timestamp for this training run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(RESULTS_DIR, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    # Load the data
    logger.info("Loading data...")
    df = pd.read_csv('df_citi_with_labels.csv')
    
    # Combine text features with proper null handling
    logger.info("Preprocessing text data...")
    text_features = [
        'Product',
        'Sub-product',
        'Issue',
        'Sub-issue',
        'Consumer complaint narrative'
    ]
    
    # Combine features with proper null handling
    df['text'] = df[text_features].fillna('').agg(' | '.join, axis=1)
    
    # Free up memory
    for col in text_features:
        df.drop(col, axis=1, inplace=True)
    gc.collect()
    
    # Take a smaller sample for initial testing
    sample_size = 10000  # Adjust based on available memory
    df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    # Split the data
    logger.info("Splitting data into train and validation sets...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(),
        df['response_label'].tolist(),
        test_size=0.2,
        random_state=42
    )
    
    # Free up memory
    del df
    gc.collect()
    
    # Initialize tokenizer and model
    logger.info("Initializing DistilBERT...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    )
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = ComplaintDataset(train_texts, train_labels, tokenizer)
    val_dataset = ComplaintDataset(val_texts, val_labels, tokenizer)
    
    # Free up memory
    del train_texts, val_texts
    gc.collect()
    
    # Initialize metrics callback
    metrics_callback = MetricsCallback()
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        logging_dir=os.path.join(run_dir, 'logs'),
        logging_steps=100,
        save_steps=500,
        eval_steps=500,
        do_eval=True
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[metrics_callback]
    )
    
    # Train the model
    logger.info("Training the model...")
    trainer.train()
    
    # Evaluate the model
    logger.info("Evaluating the model...")
    eval_results = trainer.evaluate()
    
    # Generate and save visualizations
    logger.info("Generating visualizations...")
    plot_training_curves(metrics_callback, run_dir)
    plot_confusion_matrix(
        compute_metrics.all_labels,
        compute_metrics.all_preds,
        run_dir
    )
    plot_roc_curve(
        compute_metrics.all_labels,
        compute_metrics.all_probs,
        run_dir
    )
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_callback.validation_metrics)
    metrics_df.to_csv(os.path.join(run_dir, 'training_metrics.csv'), index=False)
    
    # Save final evaluation results
    with open(os.path.join(run_dir, 'final_metrics.json'), 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    logger.info(f"Evaluation Results (saved to {run_dir}/final_metrics.json):")
    logger.info(eval_results)
    
    # Save the model and tokenizer
    logger.info("Saving the model...")
    model.save_pretrained(os.path.join(run_dir, 'model'))
    tokenizer.save_pretrained(os.path.join(run_dir, 'model'))
    logger.info(f"Model and tokenizer saved to {run_dir}/model")

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")
    raise 
import os
import json
import logging
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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

def plot_confusion_matrix(y_true, y_pred, save_path):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_prob, save_path):
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
    plt.savefig(save_path)
    plt.close()

def plot_precision_recall_curve(y_true, y_prob, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate comprehensive metrics for model evaluation"""
    report = classification_report(y_true, y_pred, output_dict=True)
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    metrics = {
        'accuracy': report['accuracy'],
        'precision_class_0': report['0']['precision'],
        'recall_class_0': report['0']['recall'],
        'f1_score_class_0': report['0']['f1-score'],
        'precision_class_1': report['1']['precision'],
        'recall_class_1': report['1']['recall'],
        'f1_score_class_1': report['1']['f1-score'],
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }
    
    return metrics

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Find the most recent model directory
    model_dirs = [d for d in os.listdir('training_results') if d.startswith('run_')]
    latest_run = max(model_dirs)
    run_dir = os.path.join('training_results', latest_run)
    model_dir = os.path.join(run_dir, 'model')  # Add model subdirectory
    
    logger.info(f'Loading model from: {model_dir}')
    
    # Load the model and tokenizer
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    
    # Load and prepare data with the same split as training
    logger.info('Loading and preparing data with the same split...')
    df = pd.read_csv('df_citi_with_labels.csv')
    
    # Combine text features with proper null handling
    text_features = [
        'Product',
        'Sub-product',
        'Issue',
        'Sub-issue',
        'Consumer complaint narrative'
    ]
    df['text'] = df[text_features].fillna('').agg(' | '.join, axis=1)
    
    # Take the same sample size with the same random state
    sample_size = 10000
    df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    # Split the data with the same random state
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(),
        df['response_label'].tolist(),
        test_size=0.2,
        random_state=42
    )
    
    # Create evaluation directory
    eval_dir = os.path.join(run_dir, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(os.path.join(eval_dir, 'plots'), exist_ok=True)
    
    # Create validation dataset and loader
    val_dataset = ComplaintDataset(val_texts, val_labels, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Evaluate
    logger.info('Starting evaluation...')
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Generating predictions"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Convert to numpy arrays
    val_predictions = np.array(all_predictions)
    val_probabilities = np.array(all_probabilities)
    val_labels = np.array(all_labels)
    
    # Calculate metrics
    metrics = calculate_metrics(val_labels, val_predictions, val_probabilities)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(eval_dir, 'evaluation_metrics.csv'), index=False)
    
    # Generate and save plots
    plot_confusion_matrix(
        val_labels,
        val_predictions,
        os.path.join(eval_dir, 'plots', 'confusion_matrix.png')
    )
    
    plot_roc_curve(
        val_labels,
        val_probabilities,
        os.path.join(eval_dir, 'plots', 'roc_curve.png')
    )
    
    plot_precision_recall_curve(
        val_labels,
        val_probabilities,
        os.path.join(eval_dir, 'plots', 'pr_curve.png')
    )
    
    # Log results
    logger.info('\nModel Evaluation Results:')
    for metric, value in metrics.items():
        logger.info(f'{metric}: {value:.3f}')
    
    logger.info(f'\nEvaluation results saved in: {eval_dir}')

if __name__ == '__main__':
    main() 
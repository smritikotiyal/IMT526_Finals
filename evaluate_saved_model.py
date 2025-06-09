import os
import json
import logging
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from enhanced_distilbert_classifier import EnhancedDistilBERTClassifier, ComplaintDataset
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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

def plot_confidence_distribution(confidences, predictions, y_true, save_path):
    plt.figure(figsize=(10, 6))
    
    correct_mask = predictions == y_true
    correct_conf = confidences[correct_mask]
    incorrect_conf = confidences[~correct_mask]
    
    plt.hist(correct_conf, alpha=0.5, label='Correct predictions', bins=50, density=True)
    plt.hist(incorrect_conf, alpha=0.5, label='Incorrect predictions', bins=50, density=True)
    
    plt.xlabel('Confidence Score')
    plt.ylabel('Density')
    plt.title('Distribution of Model Confidence Scores')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def calculate_metrics(y_true, y_pred, y_prob, confidences):
    """Calculate comprehensive metrics for model evaluation"""
    report = classification_report(y_true, y_pred, output_dict=True)
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    avg_confidence = np.mean(confidences)
    avg_confidence_correct = np.mean(confidences[y_pred == y_true])
    avg_confidence_incorrect = np.mean(confidences[y_pred != y_true])
    
    metrics = {
        'accuracy': report['accuracy'],
        'precision_class_0': report['0']['precision'],
        'recall_class_0': report['0']['recall'],
        'f1_score_class_0': report['0']['f1-score'],
        'precision_class_1': report['1']['precision'],
        'recall_class_1': report['1']['recall'],
        'f1_score_class_1': report['1']['f1-score'],
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'avg_confidence': avg_confidence,
        'avg_confidence_correct': avg_confidence_correct,
        'avg_confidence_incorrect': avg_confidence_incorrect
    }
    
    return metrics

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Find the most recent model directory
    model_dirs = [d for d in os.listdir('model_evaluation_results') if d.startswith('run_')]
    latest_run = max(model_dirs)
    run_dir = os.path.join('model_evaluation_results', latest_run)
    model_dir = os.path.join(run_dir, 'saved_model')
    
    logger.info(f'Loading model from: {model_dir}')
    
    # Load the model
    model = EnhancedDistilBERTClassifier.load_model(model_dir, device)
    model.to(device)
    model.eval()
    
    # Load the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    
    # Load and prepare data with the same split as training
    logger.info('Loading and preparing data with the same split...')
    df = pd.read_csv('df_citi_with_rag_reasoning.csv')
    
    # Prepare data (same as in training)
    df['combined_text'] = df['Product'].fillna('') + ' | ' + \
                         df['Sub-product'].fillna('') + ' | ' + \
                         df['Issue'].fillna('') + ' | ' + \
                         df['Sub-issue'].fillna('') + ' | ' + \
                         df['Consumer complaint narrative'].fillna('')
    df['response_label'] = df['response_label'].astype(int)
    
    # Recreate the exact same split using the same random state
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    
    # Create evaluation directory
    eval_dir = os.path.join(run_dir, 'evaluation_recreated_split')
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(os.path.join(eval_dir, 'plots'), exist_ok=True)
    
    # Create test dataset and loader
    test_dataset = ComplaintDataset(
        texts=test_df['combined_text'].values,
        reasonings=test_df['company_reasoning'].values,
        labels=test_df['response_label'].values,
        tokenizer=tokenizer
    )
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # Evaluate
    logger.info('Starting evaluation...')
    all_predictions = []
    all_probabilities = []
    all_confidences = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating predictions"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits, probabilities = model(input_ids, attention_mask)
            predictions = torch.argmax(probabilities, dim=1)
            confidences = torch.max(probabilities, dim=1)[0]
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    # Convert to numpy arrays
    test_predictions = np.array(all_predictions)
    test_probabilities = np.array(all_probabilities)
    test_confidences = np.array(all_confidences)
    
    # Calculate metrics
    metrics = calculate_metrics(
        test_df['response_label'].values,
        test_predictions,
        test_probabilities,
        test_confidences
    )
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(eval_dir, 'evaluation_metrics.csv'), index=False)
    
    # Generate and save plots
    plot_confusion_matrix(
        test_df['response_label'].values,
        test_predictions,
        os.path.join(eval_dir, 'plots', 'confusion_matrix.png')
    )
    
    plot_roc_curve(
        test_df['response_label'].values,
        test_probabilities,
        os.path.join(eval_dir, 'plots', 'roc_curve.png')
    )
    
    plot_precision_recall_curve(
        test_df['response_label'].values,
        test_probabilities,
        os.path.join(eval_dir, 'plots', 'pr_curve.png')
    )
    
    plot_confidence_distribution(
        test_confidences,
        test_predictions,
        test_df['response_label'].values,
        os.path.join(eval_dir, 'plots', 'confidence_distribution.png')
    )
    
    # Save detailed results
    test_results = pd.DataFrame({
        'True_Label': test_df['response_label'].values,
        'Predicted_Label': test_predictions,
        'Confidence': test_confidences,
        'Probability_Class_1': test_probabilities
    })
    test_results.to_csv(os.path.join(eval_dir, 'test_predictions.csv'), index=False)
    
    # Log results
    logger.info('\nModel Evaluation Results:')
    for metric, value in metrics.items():
        logger.info(f'{metric}: {value:.3f}')
    
    logger.info(f'\nEvaluation results saved in: {eval_dir}')

if __name__ == "__main__":
    main() 
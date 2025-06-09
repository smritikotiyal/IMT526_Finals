import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Create directories for saving results
RESULTS_DIR = 'model_evaluation_results'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'plots'), exist_ok=True)

class ComplaintDataset(Dataset):
    def __init__(self, texts, reasonings, labels, tokenizer, max_length=512):
        self.texts = texts
        self.reasonings = reasonings
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        reasoning = str(self.reasonings[idx])
        
        # Combine text and reasoning with a separator
        combined_text = text + " [SEP] " + reasoning
        
        encoding = self.tokenizer.encode_plus(
            combined_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class EnhancedDistilBERTClassifier(nn.Module):
    def __init__(self, n_classes=2, max_length=512):
        super(EnhancedDistilBERTClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, n_classes)
        self.max_length = max_length
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[0][:, 0]  # Take CLS token output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits, F.softmax(logits, dim=1)
    
    def predict(self, text):
        """Make prediction for a single text input"""
        self.eval()  # Set to evaluation mode
        
        # Tokenize the input
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Make prediction
        with torch.no_grad():
            logits, probabilities = self(inputs['input_ids'], inputs['attention_mask'])
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        return {
            'prediction': 'Yes' if prediction == 1 else 'No',
            'confidence': confidence
        }
    
    def save_model(self, save_dir):
        """Save the complete model including configuration"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model state
        torch.save(self.state_dict(), os.path.join(save_dir, 'model_state.pt'))
        
        # Save model configuration
        config = {
            'n_classes': self.classifier.out_features,
            'max_length': self.max_length,
            'dropout_rate': self.dropout.p,
            'model_type': 'distilbert-base-uncased'
        }
        
        with open(os.path.join(save_dir, 'model_config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_dir)
    
    @classmethod
    def load_model(cls, model_dir, device='cpu'):
        """Load the model from saved state and configuration"""
        # Load configuration
        with open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
            config = json.load(f)
        
        # Initialize model with saved configuration
        model = cls(
            n_classes=config['n_classes'],
            max_length=config['max_length']
        )
        
        # Load state dict
        model.load_state_dict(torch.load(
            os.path.join(model_dir, 'model_state.pt'),
            map_location=device
        ))
        
        # Load tokenizer if available
        if os.path.exists(os.path.join(model_dir, 'tokenizer_config.json')):
            model.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        
        model.to(device)
        model.eval()
        return model

def prepare_data(df):
    # Combine features into a single text
    df['combined_text'] = df['Product'].fillna('') + ' | ' + \
                         df['Sub-product'].fillna('') + ' | ' + \
                         df['Issue'].fillna('') + ' | ' + \
                         df['Sub-issue'].fillna('') + ' | ' + \
                         df['Consumer complaint narrative'].fillna('')
    
    # Convert labels to numeric
    df['response_label'] = df['response_label'].astype(int)
    
    return df

def train_model(model, train_loader, val_loader, device, num_epochs=3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    best_val_accuracy = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item())})
        
        avg_train_loss = total_loss / len(train_loader)
        val_accuracy = evaluate_model(model, val_loader, device)
        
        logger.info(f'Epoch {epoch + 1}: Average training loss = {avg_train_loss:.3f}')
        logger.info(f'Epoch {epoch + 1}: Validation accuracy = {val_accuracy:.3f}')
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # Save the model
            torch.save(model.state_dict(), 'enhanced_complaint_classifier.pt')
            logger.info(f'Model saved with validation accuracy: {val_accuracy:.3f}')

def evaluate_model(model, data_loader, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits, _ = model(input_ids, attention_mask)
            _, predictions = torch.max(logits, dim=1)
            
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.shape[0]
    
    return correct_predictions / total_predictions

def predict_with_confidence(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_confidences = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits, probabilities = model(input_ids, attention_mask)
            predictions = torch.argmax(probabilities, dim=1)
            confidences = torch.max(probabilities, dim=1)[0]
            
            all_predictions.extend(predictions.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_confidences)

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
    
    # Separate confidences for correct and incorrect predictions
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
    # Basic classification metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Calculate ROC AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Calculate PR AUC
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    # Calculate confidence metrics
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
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(RESULTS_DIR, f'run_{timestamp}')
    os.makedirs(run_dir)
    os.makedirs(os.path.join(run_dir, 'plots'))
    model_save_dir = os.path.join(run_dir, 'saved_model')
    os.makedirs(model_save_dir)
    
    # Load and prepare data
    logger.info('Loading data...')
    df = pd.read_csv('df_citi_with_rag_reasoning.csv')
    df = prepare_data(df)
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    
    # Save the data splits
    logger.info('Saving data splits...')
    data_splits_dir = os.path.join(run_dir, 'data_splits')
    os.makedirs(data_splits_dir)
    train_df.to_csv(os.path.join(data_splits_dir, 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(data_splits_dir, 'val_split.csv'), index=False)
    test_df.to_csv(os.path.join(data_splits_dir, 'test_split.csv'), index=False)
    
    # Save split indices for reproducibility
    split_info = {
        'train_indices': train_df.index.tolist(),
        'val_indices': val_df.index.tolist(),
        'test_indices': test_df.index.tolist(),
        'random_state': 42
    }
    with open(os.path.join(data_splits_dir, 'split_indices.json'), 'w') as f:
        json.dump(split_info, f)
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Save tokenizer
    tokenizer.save_pretrained(model_save_dir)
    
    # Create datasets and loaders
    train_dataset = ComplaintDataset(
        texts=train_df['combined_text'].values,
        reasonings=train_df['company_reasoning'].values,
        labels=train_df['response_label'].values,
        tokenizer=tokenizer
    )
    
    val_dataset = ComplaintDataset(
        texts=val_df['combined_text'].values,
        reasonings=val_df['company_reasoning'].values,
        labels=val_df['response_label'].values,
        tokenizer=tokenizer
    )
    
    test_dataset = ComplaintDataset(
        texts=test_df['combined_text'].values,
        reasonings=test_df['company_reasoning'].values,
        labels=test_df['response_label'].values,
        tokenizer=tokenizer
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # Initialize and train model
    model = EnhancedDistilBERTClassifier()
    model.to(device)
    
    logger.info('Starting training...')
    train_model(model, train_loader, val_loader, device)
    
    # Save the final model
    logger.info('Saving the final model...')
    model.save_model(model_save_dir)
    
    # Save training configuration
    training_config = {
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'batch_size': 16,
        'learning_rate': 2e-5,
        'num_epochs': 3,
        'device': str(device),
        'timestamp': timestamp
    }
    
    with open(os.path.join(model_save_dir, 'training_config.json'), 'w') as f:
        json.dump(training_config, f, indent=4)
    
    # Evaluate on test set
    logger.info('Evaluating on test set...')
    model.eval()
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
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of class 1
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
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(run_dir, 'metrics.csv'), index=False)
    
    # Generate and save plots
    plot_confusion_matrix(
        test_df['response_label'].values,
        test_predictions,
        os.path.join(run_dir, 'plots', 'confusion_matrix.png')
    )
    
    plot_roc_curve(
        test_df['response_label'].values,
        test_probabilities,
        os.path.join(run_dir, 'plots', 'roc_curve.png')
    )
    
    plot_precision_recall_curve(
        test_df['response_label'].values,
        test_probabilities,
        os.path.join(run_dir, 'plots', 'pr_curve.png')
    )
    
    plot_confidence_distribution(
        test_confidences,
        test_predictions,
        test_df['response_label'].values,
        os.path.join(run_dir, 'plots', 'confidence_distribution.png')
    )
    
    # Save detailed results
    test_results = pd.DataFrame({
        'True_Label': test_df['response_label'].values,
        'Predicted_Label': test_predictions,
        'Confidence': test_confidences,
        'Probability_Class_1': test_probabilities
    })
    test_results.to_csv(os.path.join(run_dir, 'test_predictions.csv'), index=False)
    
    # Log results
    logger.info('\nModel Evaluation Results:')
    for metric, value in metrics.items():
        logger.info(f'{metric}: {value:.3f}')
    
    logger.info(f'\nResults saved in: {run_dir}')
    logger.info(f'Model saved in: {model_save_dir}')

if __name__ == "__main__":
    main() 
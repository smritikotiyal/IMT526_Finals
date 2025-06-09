import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplaintClassifier:
    def __init__(self, model_path='./complaint_classifier_model'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        logger.info("Loading tokenizer and model...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
    def preprocess_text(self, product, sub_product, issue, sub_issue, narrative):
        """Combine features in the same format as training data"""
        features = [
            str(product) if product else '',
            str(sub_product) if sub_product else '',
            str(issue) if issue else '',
            str(sub_issue) if sub_issue else '',
            str(narrative) if narrative else ''
        ]
        return ' | '.join(features)
    
    def predict(self, product, sub_product, issue, sub_issue, narrative):
        """
        Make prediction for a single complaint
        Returns:
        - prediction (int): 0 (non-monetary relief) or 1 (monetary relief)
        - probability (float): confidence score for the prediction
        """
        # Preprocess input
        text = self.preprocess_text(product, sub_product, issue, sub_issue, narrative)
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        return prediction, confidence

    def predict_batch(self, complaints):
        """
        Make predictions for a batch of complaints
        complaints: list of dicts, each containing 'Product', 'Sub-product', 'Issue', 'Sub-issue', 'Consumer complaint narrative'
        Returns:
        - predictions (list): list of 0s and 1s
        - probabilities (list): confidence scores for each prediction
        """
        texts = [
            self.preprocess_text(
                c.get('Product', ''),
                c.get('Sub-product', ''),
                c.get('Issue', ''),
                c.get('Sub-issue', ''),
                c.get('Consumer complaint narrative', '')
            )
            for c in complaints
        ]
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1).tolist()
            confidences = [probabilities[i][pred].item() for i, pred in enumerate(predictions)]
        
        return predictions, confidences

# Example usage
if __name__ == "__main__":
    # Initialize classifier
    classifier = ComplaintClassifier()
    
    # Example single prediction
    example_complaint = {
        'Product': 'Credit card',
        'Sub-product': 'General-purpose credit card',
        'Issue': 'Problem with a purchase shown on your statement',
        'Sub-issue': 'Credit card purchase shown was not made',
        'Consumer complaint narrative': 'I found unauthorized charges on my credit card statement.'
    }
    
    prediction, confidence = classifier.predict(
        example_complaint['Product'],
        example_complaint['Sub-product'],
        example_complaint['Issue'],
        example_complaint['Sub-issue'],
        example_complaint['Consumer complaint narrative']
    )
    
    logger.info("\nSingle Prediction Example:")
    logger.info(f"Text: {example_complaint['Consumer complaint narrative']}")
    logger.info(f"Prediction: {'Monetary Relief' if prediction == 1 else 'Non-monetary Relief'}")
    logger.info(f"Confidence: {confidence:.2%}")
    
    # Example batch prediction
    example_complaints = [
        example_complaint,
        {
            'Product': 'Mortgage',
            'Sub-product': 'Conventional fixed mortgage',
            'Issue': 'Loan servicing, payments, escrow account',
            'Sub-issue': 'Payment processing',
            'Consumer complaint narrative': 'My mortgage payment was not properly applied to my account.'
        }
    ]
    
    predictions, confidences = classifier.predict_batch(example_complaints)
    
    logger.info("\nBatch Prediction Example:")
    for i, (pred, conf) in enumerate(zip(predictions, confidences)):
        logger.info(f"\nComplaint {i+1}:")
        logger.info(f"Prediction: {'Monetary Relief' if pred == 1 else 'Non-monetary Relief'}")
        logger.info(f"Confidence: {conf:.2%}") 
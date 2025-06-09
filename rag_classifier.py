import os
import torch
import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from transformers import DistilBertTokenizer, DistilBertModel
import logging
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
from inference import ComplaintClassifier
import PyPDF2
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    content: str
    metadata: dict

class PolicyRetriever:
    def __init__(self, policy_dir: str, embedding_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.policy_dir = Path(policy_dir)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.document_store: List[Document] = []
        self.index = None
        self.initialize_faiss()
    
    def _read_pdf(self, file_path: Path) -> str:
        """Read content from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                # Create PDF reader object
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from all pages
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                # Clean the extracted text
                text = re.sub(r'\s+', ' ', text).strip()  # Remove excessive whitespace
                return text
        except Exception as e:
            logger.error(f"Error reading PDF file {file_path}: {str(e)}")
            return ""
    
    def initialize_faiss(self):
        """Initialize FAISS index and load policy documents"""
        logger.info("Initializing FAISS index and loading policy documents...")
        
        # Load and process all policy documents
        self._load_policy_documents()
        
        if not self.document_store:
            logger.error("No policy documents found or successfully loaded!")
            raise ValueError("No policy documents available for indexing")
        
        # Create FAISS index
        embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(embedding_dim)
        
        # Get embeddings for all documents
        embeddings = self.embedding_model.encode(
            [doc.content for doc in self.document_store],
            convert_to_tensor=False
        )
        self.index.add(np.array(embeddings))
        logger.info(f"Successfully indexed {len(self.document_store)} policy documents")
    
    def _load_policy_documents(self):
        """Load all policy documents from the policy directory"""
        if not self.policy_dir.exists():
            raise ValueError(f"Policy directory {self.policy_dir} does not exist!")
        
        pdf_files = list(self.policy_dir.rglob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in {self.policy_dir}")
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        for file_path in pdf_files:
            content = self._read_pdf(file_path)
            if content:  # Only add if content was successfully extracted
                doc = Document(
                    content=content,
                    metadata={
                        "file_name": file_path.name,
                        "file_path": str(file_path)
                    }
                )
                self.document_store.append(doc)
                logger.info(f"Successfully loaded: {file_path.name}")
            else:
                logger.warning(f"Skipped {file_path.name} due to extraction error")
    
    def retrieve_relevant_policies(self, query: str, k: int = 3) -> List[Document]:
        """Retrieve k most relevant policy documents for the query"""
        if not self.document_store:
            raise ValueError("No policy documents available")
        
        # Adjust k if it's larger than the number of available documents
        k = min(k, len(self.document_store))
        
        # Get query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, k)
        
        # Return relevant documents
        return [self.document_store[i] for i in indices[0]]

class RAGComplaintClassifier:
    def __init__(
        self,
        policy_dir: str,
        model_path: str = './training_results/run_20250530_174716/model',
        llm_api_key: Optional[str] = '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',
        # llm_api_key: Optional[str] = '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',
        llm_provider: str = 'openai',
        max_policy_chars: int = 1000  # Maximum characters per policy
    ):
        self.max_policy_chars = max_policy_chars
        # Initialize policy retriever
        self.policy_retriever = PolicyRetriever(policy_dir)
        
        # Initialize ML classifier
        self.ml_classifier = ComplaintClassifier(model_path)
        
        # Set up LLM
        self.llm_provider = llm_provider
        self.llm_api_key = llm_api_key or os.getenv(f"{llm_provider.upper()}_API_KEY")
        if not self.llm_api_key:
            raise ValueError(f"No API key provided for {llm_provider}")
        
        self._setup_llm()
    
    def _setup_llm(self):
        """Setup LLM based on provider"""
        if self.llm_provider == 'openai':
            from openai import OpenAI
            self.llm_client = OpenAI(api_key=self.llm_api_key)
        elif self.llm_provider == 'anthropic':
            import anthropic
            self.llm_client = anthropic.Anthropic(api_key=self.llm_api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM"""
        if self.llm_provider == 'openai':
            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes customer complaints and relevant policies to determine if monetary relief is warranted."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        elif self.llm_provider == 'anthropic':
            response = self.llm_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
    
    def _format_complaint(self, complaint: Dict[str, str]) -> str:
        """Format complaint for query and prompt"""
        return f"""
Product: {complaint.get('Product', '')}
Sub-product: {complaint.get('Sub-product', '')}
Issue: {complaint.get('Issue', '')}
Sub-issue: {complaint.get('Sub-issue', '')}
Complaint: {complaint.get('Consumer complaint narrative', '')}
"""
    
    def _truncate_policy(self, policy_text: str) -> str:
        """Truncate policy text to max_policy_chars"""
        if len(policy_text) <= self.max_policy_chars:
            return policy_text
        return policy_text[:self.max_policy_chars] + "..."
    
    def classify_with_reasoning(self, complaint: Dict[str, str]) -> Dict:
        """
        Classify complaint using both ML model and policy-based reasoning
        Returns dict with classification, confidence, and reasoning
        """
        # Get ML model prediction
        ml_prediction, ml_confidence = self.ml_classifier.predict(
            complaint.get('Product', ''),
            complaint.get('Sub-product', ''),
            complaint.get('Issue', ''),
            complaint.get('Sub-issue', ''),
            complaint.get('Consumer complaint narrative', '')
        )
        
        # Format complaint for retrieval
        formatted_complaint = self._format_complaint(complaint)
        
        # Retrieve relevant policies (limit to 2 most relevant)
        relevant_policies = self.policy_retriever.retrieve_relevant_policies(formatted_complaint, k=2)
        
        # Create prompt for LLM with truncated policies
        prompt = f"""
Based on the following customer complaint and relevant policies, determine if monetary relief is warranted.
Provide your reasoning and final decision (0 for non-monetary relief, 1 for monetary relief).

COMPLAINT:
{formatted_complaint}

RELEVANT POLICIES:
{chr(10).join(f"Policy {i+1}:{chr(10)}{self._truncate_policy(doc.content)}{chr(10)}" for i, doc in enumerate(relevant_policies))}

ML MODEL PREDICTION:
The ML model predicts: {"Monetary Relief" if ml_prediction == 1 else "Non-monetary Relief"} with {ml_confidence:.2%} confidence.

Please analyze the complaint and policies, and provide:
1. Your reasoning for or against monetary relief
2. Your final decision (0 or 1)
3. Your confidence in the decision (0-1)

Format your response as JSON with keys: "reasoning", "decision", "confidence"
"""
        
        # Get LLM response
        llm_response = self._get_llm_response(prompt)
        
        try:
            llm_result = json.loads(llm_response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM response as JSON: {llm_response}")
            llm_result = {
                "reasoning": "Failed to parse LLM response",
                "decision": ml_prediction,
                "confidence": ml_confidence
            }
        
        # Combine ML and LLM results
        final_result = {
            "ml_prediction": ml_prediction,
            "ml_confidence": ml_confidence,
            "llm_reasoning": llm_result.get("reasoning", ""),
            "llm_decision": llm_result.get("decision", ml_prediction),
            "llm_confidence": llm_result.get("confidence", ml_confidence),
            "final_decision": llm_result.get("decision", ml_prediction)  # Use LLM decision as final
        }
        
        return final_result

    def evaluate(self, test_data: List[Dict[str, str]], true_labels: List[int]) -> Dict[str, float]:
        """
        Evaluate the RAG classifier on test data
        Args:
            test_data: List of complaints in dictionary format
            true_labels: List of true labels (0 or 1)
        Returns:
            Dictionary containing evaluation metrics
        """
        ml_predictions = []
        llm_predictions = []
        ml_confidences = []
        llm_confidences = []
        llm_reasoning_list = []
        
        for complaint in test_data:
            result = self.classify_with_reasoning(complaint)
            ml_predictions.append(result['ml_prediction'])
            llm_predictions.append(result['llm_decision'])
            ml_confidences.append(result['ml_confidence'])
            llm_confidences.append(result['llm_confidence'])
            llm_reasoning_list.append(result['llm_reasoning'])
        
        # Calculate metrics for ML model
        ml_metrics = {
            'ml_accuracy': accuracy_score(true_labels, ml_predictions),
            'ml_precision': precision_score(true_labels, ml_predictions),
            'ml_recall': recall_score(true_labels, ml_predictions),
            'ml_f1': f1_score(true_labels, ml_predictions),
            'ml_avg_confidence': np.mean(ml_confidences)
        }
        
        # Calculate metrics for LLM/RAG
        llm_metrics = {
            'llm_accuracy': accuracy_score(true_labels, llm_predictions),
            'llm_precision': precision_score(true_labels, llm_predictions),
            'llm_recall': recall_score(true_labels, llm_predictions),
            'llm_f1': f1_score(true_labels, llm_predictions),
            'llm_avg_confidence': np.mean(llm_confidences)
        }
        
        # Calculate agreement metrics
        agreement = np.mean([1 if ml == llm else 0 for ml, llm in zip(ml_predictions, llm_predictions)])
        
        # Combine all metrics
        metrics = {
            **ml_metrics,
            **llm_metrics,
            'model_agreement': agreement
        }
        
        return metrics

# Example usage
if __name__ == "__main__":
    # Initialize classifier
    classifier = RAGComplaintClassifier(
        policy_dir="Citibank_Policies",
        llm_provider="openai"
    )
    
    # Load test data
    df = pd.read_csv('df_citi_with_labels.csv')
    test_size = 10  # Small test set for demonstration
    test_data = df.sample(n=test_size, random_state=42).to_dict('records')
    true_labels = df['response_label'].values[:test_size]
    
    # Run evaluation
    logger.info("\nRunning evaluation on test set...")
    metrics = classifier.evaluate(test_data, true_labels)
    
    # Print evaluation results
    logger.info("\nEvaluation Results:")
    logger.info("\nML Model Metrics:")
    logger.info(f"Accuracy: {metrics['ml_accuracy']:.2%}")
    logger.info(f"Precision: {metrics['ml_precision']:.2%}")
    logger.info(f"Recall: {metrics['ml_recall']:.2%}")
    logger.info(f"F1 Score: {metrics['ml_f1']:.2%}")
    logger.info(f"Average Confidence: {metrics['ml_avg_confidence']:.2%}")
    
    logger.info("\nLLM/RAG Metrics:")
    logger.info(f"Accuracy: {metrics['llm_accuracy']:.2%}")
    logger.info(f"Precision: {metrics['llm_precision']:.2%}")
    logger.info(f"Recall: {metrics['llm_recall']:.2%}")
    logger.info(f"F1 Score: {metrics['llm_f1']:.2%}")
    logger.info(f"Average Confidence: {metrics['llm_avg_confidence']:.2%}")
    
    logger.info(f"\nModel Agreement: {metrics['model_agreement']:.2%}")
    
    # Example complaint for individual prediction
    example_complaint = {
        'Product': 'Credit card',
        'Sub-product': 'General-purpose credit card',
        'Issue': 'Problem with a purchase shown on your statement',
        'Sub-issue': 'Credit card purchase shown was not made',
        'Consumer complaint narrative': 'I found unauthorized charges on my credit card statement.'
    }
    
    # Get classification with reasoning
    result = classifier.classify_with_reasoning(example_complaint)
    
    # Print individual result
    logger.info("\nExample Complaint Classification:")
    logger.info(f"ML Model Prediction: {'Monetary Relief' if result['ml_prediction'] == 1 else 'Non-monetary Relief'}")
    logger.info(f"ML Confidence: {result['ml_confidence']:.2%}")
    logger.info("\nLLM Analysis:")
    logger.info(f"Reasoning: {result['llm_reasoning']}")
    logger.info(f"Decision: {'Monetary Relief' if result['llm_decision'] == 1 else 'Non-monetary Relief'}")
    logger.info(f"Confidence: {result['llm_confidence']:.2%}")
    logger.info(f"\nFinal Decision: {'Monetary Relief' if result['final_decision'] == 1 else 'Non-monetary Relief'}") 
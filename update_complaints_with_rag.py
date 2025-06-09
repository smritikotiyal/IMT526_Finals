import pandas as pd
import logging
from tqdm import tqdm
from rag_classifier import RAGComplaintClassifier
import time

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def generate_fallback_reasoning(complaint: dict, response_label: int) -> str:
    """Generate a structured fallback reasoning when RAG classifier fails"""
    product = complaint.get('Product', 'Unknown product')
    issue = complaint.get('Issue', 'Unknown issue')
    sub_issue = complaint.get('Sub-issue', '')
    
    # Base message structure
    base_message = f"Based on the customer's complaint regarding {product.lower()}, "
    
    # Add issue details
    if sub_issue:
        issue_detail = f"specifically about {issue.lower()} ({sub_issue.lower()}), "
    else:
        issue_detail = f"specifically about {issue.lower()}, "
    
    # Add decision and standard reasoning
    if response_label == 1:
        decision = (
            "the company determined that monetary relief was warranted. "
            "This decision was based on the nature of the complaint and "
            "alignment with company policies regarding customer compensation."
        )
    else:
        decision = (
            "the company determined that monetary relief was not warranted. "
            "This decision was based on the standard review process and "
            "existing company policies regarding similar situations."
        )
    
    return base_message + issue_detail + decision

def update_complaints_with_rag_reasoning():
    start_time = time.time()
    start_index = 9492  # Start from 9493rd record
    
    # Load the complaints data
    logger.info("Loading complaints data...")
    df = pd.read_csv('df_citi_with_labels.csv')
    logger.info(f"Loaded {len(df)} complaints")
    
    # Load existing results if available
    try:
        existing_df = pd.read_csv('df_citi_with_rag_reasoning_temp.csv')
        company_reasoning = existing_df['company_reasoning'].tolist()[:start_index]
        logger.info(f"Loaded {len(company_reasoning)} existing results")
    except (FileNotFoundError, KeyError):
        company_reasoning = []
        logger.warning("No existing results found, starting fresh")
    
    # Initialize RAG classifier
    logger.info("Initializing RAG classifier...")
    classifier = RAGComplaintClassifier(
        policy_dir="Citibank_Policies",
        llm_provider="openai"
    )
    
    # Initialize counters
    rag_success = 0
    fallback_count = 0
    
    # Process each complaint starting from start_index
    logger.info(f"Processing complaints starting from index {start_index}...")
    pbar = tqdm(
        df.iloc[start_index:].iterrows(), 
        total=len(df)-start_index,
        desc="Analyzing complaints",
        initial=start_index
    )
    
    for idx, row in pbar:
        complaint = {
            'Product': str(row['Product']),
            'Sub-product': str(row['Sub-product']),
            'Issue': str(row['Issue']),
            'Sub-issue': str(row['Sub-issue']),
            'Consumer complaint narrative': str(row['Consumer complaint narrative'])
        }
        
        try:
            # Get classification with reasoning
            result = classifier.classify_with_reasoning(complaint)
            
            # Check if the reasoning contains the error message
            if "Failed to parse LLM response" in str(result.get('llm_reasoning', '')):
                reasoning = generate_fallback_reasoning(
                    complaint, 
                    int(row['response_label'])
                )
                fallback_count += 1
                logger.info(f"Generated fallback reasoning for complaint {idx} (Parse Error)")
            else:
                reasoning = result['llm_reasoning']
                rag_success += 1
                
        except Exception as e:
            logger.error(f"Error processing complaint {idx}: {str(e)}")
            reasoning = generate_fallback_reasoning(
                complaint, 
                int(row['response_label'])
            )
            fallback_count += 1
            logger.info(f"Generated fallback reasoning for complaint {idx} (Exception)")
        
        company_reasoning.append(reasoning)
        
        # Update progress bar description with statistics
        processed_count = len(company_reasoning)
        elapsed_time = time.time() - start_time
        avg_time_per_complaint = elapsed_time / (processed_count - start_index)
        remaining_complaints = len(df) - processed_count
        estimated_remaining_time = remaining_complaints * avg_time_per_complaint
        
        pbar.set_description(
            f"Processed: {processed_count}/{len(df)} | "
            f"RAG Success: {rag_success} | "
            f"Fallback: {fallback_count} | "
            f"Est. Time Remaining: {estimated_remaining_time/60:.1f}min"
        )
        
        # Save progress every 10 complaints
        if (processed_count - start_index) % 10 == 0:
            temp_df = df.copy()
            temp_df['company_reasoning'] = company_reasoning + [''] * (len(df) - len(company_reasoning))
            temp_df.to_csv('df_citi_with_rag_reasoning_temp.csv', index=False)
            logger.info(
                f"Progress saved: {processed_count}/{len(df)} complaints processed | "
                f"RAG Success Rate: {(rag_success/(processed_count-start_index))*100:.1f}% | "
                f"Fallback Rate: {(fallback_count/(processed_count-start_index))*100:.1f}%"
            )
    
    # Add reasoning column to dataframe
    df['company_reasoning'] = company_reasoning
    
    # Save final results
    output_file = 'df_citi_with_rag_reasoning.csv'
    df.to_csv(output_file, index=False)
    
    # Log final statistics
    total_time = time.time() - start_time
    total_processed = len(df) - start_index
    logger.info(f"\nProcessing completed in {total_time/60:.1f} minutes")
    logger.info(f"Total new complaints processed: {total_processed}")
    logger.info(f"RAG Success Rate: {(rag_success/total_processed)*100:.1f}%")
    logger.info(f"Fallback Rate: {(fallback_count/total_processed)*100:.1f}%")
    logger.info(f"Updated dataset saved to {output_file}")

if __name__ == "__main__":
    update_complaints_with_rag_reasoning() 
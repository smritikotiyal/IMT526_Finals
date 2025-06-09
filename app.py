from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from enhanced_distilbert_classifier import EnhancedDistilBERTClassifier
import torch
from rag_pipeline import RAGPipeline
from dotenv import load_dotenv

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)
app.secret_key = 'citibank_complaint_system'

# Constants
TICKETS_FILE = 'tickets.csv'
SLA_DAYS = 15

# Initialize models
print("Loading models...")
distilbert_dir = os.path.join('training_results', 'run_20250530_174716', 'model')
distilbert = DistilBertForSequenceClassification.from_pretrained(distilbert_dir)
tokenizer = DistilBertTokenizer.from_pretrained(distilbert_dir)

enhanced_dir = os.path.join('model_evaluation_results', 'run_20250601_152655', 'saved_model')
enhanced_model = EnhancedDistilBERTClassifier.load_model(enhanced_dir)
print("Models loaded successfully!")

def get_tickets():
    if not os.path.exists(TICKETS_FILE):
        return pd.DataFrame(columns=['ticket_id', 'created_timestamp', 'status', 
                                   'product', 'sub_product', 'issue', 'sub_issue', 
                                   'complaint_narrative', 'distilbert_prediction',
                                   'distilbert_confidence', 'enhanced_prediction',
                                   'enhanced_confidence', 'enhanced_reasoning',
                                   'final_decision', 'employee_reasoning',
                                   'closed_timestamp', 'sla_status', 'sources'])
    return pd.read_csv(TICKETS_FILE)

def update_sla_status():
    tickets = get_tickets()
    if len(tickets) == 0:
        return tickets
    
    now = datetime.now()
    tickets['created_timestamp'] = pd.to_datetime(tickets['created_timestamp'])
    
    # Calculate days since creation for open tickets
    mask = tickets['status'] != 'Closed'
    tickets.loc[mask, 'days_pending'] = (now - tickets.loc[mask, 'created_timestamp']).dt.days
    
    # Update SLA status
    tickets.loc[mask & (tickets['days_pending'] >= SLA_DAYS), 'sla_status'] = 'Overdue'
    tickets.loc[mask & (tickets['days_pending'] >= SLA_DAYS * 0.8) & (tickets['days_pending'] < SLA_DAYS), 'sla_status'] = 'Warning'
    tickets.loc[mask & (tickets['days_pending'] < SLA_DAYS * 0.8), 'sla_status'] = 'On Track'
    
    return tickets

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_complaint', methods=['POST'])
def submit_complaint():
    ticket_data = {
        'ticket_id': str(uuid.uuid4()),
        'created_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'status': 'Open',
        'product': request.form['product'],
        'sub_product': request.form['sub_product'],
        'issue': request.form['issue'],
        'sub_issue': request.form['sub_issue'],
        'complaint_narrative': request.form['complaint_narrative'],
        'distilbert_prediction': '',
        'distilbert_confidence': '',
        'enhanced_prediction': '',
        'enhanced_confidence': '',
        'enhanced_reasoning': '',
        'final_decision': '',
        'employee_reasoning': '',
        'closed_timestamp': '',
        'sla_status': 'On Track'
    }
    
    tickets = get_tickets()
    tickets = pd.concat([tickets, pd.DataFrame([ticket_data])], ignore_index=True)
    tickets.to_csv(TICKETS_FILE, index=False)
    
    flash('Complaint submitted successfully! Your ticket ID is: ' + ticket_data['ticket_id'])
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    tickets = update_sla_status()
    return render_template('dashboard.html', tickets=tickets.to_dict('records'))

@app.route('/ticket/<ticket_id>')
def ticket_details(ticket_id):
    tickets = get_tickets()
    ticket = tickets[tickets['ticket_id'] == ticket_id].iloc[0].to_dict()
    return render_template('ticket_details.html', ticket=ticket)

@app.route('/assess_ticket/<ticket_id>', methods=['POST'])
def assess_ticket(ticket_id):
    tickets = get_tickets()
    ticket_idx = tickets.index[tickets['ticket_id'] == ticket_id][0]
    ticket = tickets.iloc[ticket_idx]
    
    # Prepare input text
    input_text = f"{ticket['product']} | {ticket['sub_product']} | {ticket['issue']} | {ticket['sub_issue']} | {ticket['complaint_narrative']}"
    
    # Get DistilBERT prediction
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding=True)
    outputs = distilbert(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probs, dim=1).item()
    confidence = probs[0][prediction].item()
    
    # Get Enhanced DistilBERT prediction
    enhanced_result = enhanced_model.predict(input_text)
    
    # Update ticket
    tickets.at[ticket_idx, 'status'] = 'Assessed'
    tickets.at[ticket_idx, 'distilbert_prediction'] = 'Monetary Relief' if prediction == 1 else 'Non-monetary Relief'
    tickets.at[ticket_idx, 'distilbert_confidence'] = f"{confidence:.2%}"
    tickets.at[ticket_idx, 'enhanced_prediction'] = enhanced_result['prediction']
    tickets.at[ticket_idx, 'enhanced_confidence'] = f"{enhanced_result['confidence']:.2%}"
    
    rag = RAGPipeline()
    chain = rag.get_chain()
    
    # Test query
    question = ticket['complaint_narrative']
    try:
        result = rag.query(question)
        '''print(f"\nQuestion: {question}")
        print(f"\nAnswer: {result['answer']}")
        print("\nSources:")'''
        tickets.at[ticket_idx, 'enhanced_reasoning'] = result['answer']
        tickets.at[ticket_idx, 'sources'] = ""
        # tickets.at[ticket_idx, 'sources'] = result['source_documents']
       # print('result[source_documents] : ', result['source_documents'])
        for doc in result['source_documents']:
            print(f"- {doc.metadata['source']}, page {doc.metadata['page']}")
            tickets.at[ticket_idx, 'sources'] += f"- {doc.metadata['source']}, page {doc.metadata['page']}\n"
            print('tickets.at[ticket_idx, sources] : ', tickets.at[ticket_idx, 'sources'])
    except Exception as e:
        print(f"Error: {str(e)}") 

    tickets.to_csv(TICKETS_FILE, index=False)
    return redirect(url_for('ticket_details', ticket_id=ticket_id))

@app.route('/resolve_ticket/<ticket_id>', methods=['POST'])
def resolve_ticket(ticket_id):
    tickets = get_tickets()
    ticket_idx = tickets.index[tickets['ticket_id'] == ticket_id][0]
    
    action = request.form['action']
    if action == 'accept':
        # Use model's recommendation
        tickets.at[ticket_idx, 'final_decision'] = tickets.at[ticket_idx, 'enhanced_prediction']
        tickets.at[ticket_idx, 'employee_reasoning'] = tickets.at[ticket_idx, 'enhanced_reasoning']
    else:
        # Use employee's input
        tickets.at[ticket_idx, 'final_decision'] = request.form['decision']
        tickets.at[ticket_idx, 'employee_reasoning'] = request.form['reasoning']
    
    tickets.at[ticket_idx, 'status'] = 'Closed'
    tickets.at[ticket_idx, 'closed_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    tickets.to_csv(TICKETS_FILE, index=False)
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    app.run(debug=True) 
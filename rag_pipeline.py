import os
from typing import List, Dict
import logging
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, policy_dir: str = "Citibank_Policies"):
        """
        Initialize the RAG Pipeline with policy documents.
        
        Args:
            policy_dir (str): Directory containing policy PDF documents
        """
        self.policy_dir = policy_dir
        self.documents = []
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Create vector store directory if it doesn't exist
        vector_store_dir = "vector_store"
        if not os.path.exists(vector_store_dir):
            os.makedirs(vector_store_dir)
        
        # Initialize the pipeline
        self._load_documents()
        
        # Check if vector store files exist
        index_file = os.path.join(vector_store_dir, "index.faiss")
        docstore_file = os.path.join(vector_store_dir, "docstore.pkl")
        
        if os.path.exists(index_file) and os.path.exists(docstore_file):
            logger.info("Loading existing vector store...")
            self.vector_store = FAISS.load_local(
                folder_path=vector_store_dir,
                embeddings=self.embeddings,
                index_name="index"
            )
            logger.info("Vector store loaded successfully")
        else:
            logger.info("Creating new vector store...")
            self._create_vector_store()
            
        self._setup_llm()
        self._create_qa_chain()
        
    def _load_documents(self):
        """Load and process PDF documents from the policy directory."""
        logger.info(f"Loading documents from {self.policy_dir}")
        
        # Get all PDF files in the directory
        pdf_files = [f for f in os.listdir(self.policy_dir) if f.endswith('.pdf')]
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        # Load each PDF and split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        for pdf_file in pdf_files:
            file_path = os.path.join(self.policy_dir, pdf_file)
            try:
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                chunks = text_splitter.split_documents(pages)
                self.documents.extend(chunks)
                logger.info(f"Successfully processed: {pdf_file}")
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")
        
        logger.info(f"Total chunks created: {len(self.documents)}")

    def _create_vector_store(self):
        """Create FAISS vector store from processed documents."""
        logger.info("Creating FAISS vector store")
        
        # Create vector store using the embeddings initialized in __init__
        self.vector_store = FAISS.from_documents(
            documents=self.documents,
            embedding=self.embeddings
        )
        
        # Save the vector store
        self.vector_store.save_local("vector_store", "index")
        logger.info("Vector store created and saved successfully")

    def _setup_llm(self):
        """Initialize the GPT-3.5 model."""
        logger.info("Setting up GPT-3.5 model")
        
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=500
        )

    def _create_qa_chain(self):
        """Create the question-answering chain."""
        logger.info("Creating QA chain")
        
        # Create a conversation memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Custom prompt template
        prompt_template = """You are a Citibank complaint resolution specialist. Use the following pieces of context to 
        answer questions about Citibank's policies and procedures. If you don't know the answer, just say that you don't 
        know, don't try to make up an answer.

        Context: {context}

        Chat History: {chat_history}
        
        Question: {question}

        Answer:"""

        PROMPT = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=prompt_template
        )
        
        # Create the chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 3}
            ),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True,
            verbose=True
        )
        
        logger.info("QA chain created successfully")

    def get_chain(self) -> ConversationalRetrievalChain:
        """
        Get the QA chain for use in the application.
        
        Returns:
            ConversationalRetrievalChain: The initialized QA chain
        """
        return self.qa_chain

    def query(self, question: str) -> Dict:
        """
        Query the RAG pipeline with a question.
        
        Args:
            question (str): The question to ask
            
        Returns:
            Dict: Contains 'answer' and 'source_documents'
        """
        if not self.qa_chain:
            raise ValueError("QA chain not initialized")
        
        try:
            result = self.qa_chain({"question": question})
            #print(result)
            return {
                'answer': result['answer'],
                'source_documents': result['source_documents']
            }
        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    rag = RAGPipeline()
    chain = rag.get_chain()
    
    # Test query
    question = "What is Citibank's policy on credit card dispute resolution?"
    try:
        result = rag.query(question)
        # print(f"\nQuestion: {question}")
        # print(f"\nAnswer: {result['answer']}")
        # print("\nSources:")
        for doc in result['source_documents']:
            print(f"- {doc.metadata['source']}, page {doc.metadata['page']}")
    except Exception as e:
        print(f"Error: {str(e)}") 
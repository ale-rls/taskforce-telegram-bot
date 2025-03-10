from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
import os

# Pinecone credentials
# UPDATE CREDENTIALS
# PINECONE_API_KEY = ""
INDEX_NAME = "telegram-bot-docs"

print("Initializing Pinecone")
# Initialize Pinecone with the new API
pc = Pinecone(api_key=PINECONE_API_KEY)

# List available indexes
print("Available indexes:")
print(pc.list_indexes())

# Connect to your index
try:
    index = pc.Index(INDEX_NAME)
    print(f"Successfully connected to index: {INDEX_NAME}")
    
    # Check stats
    stats = index.describe_index_stats()
    print(f"Index stats: {stats}")
except Exception as e:
    print(f"Error connecting to index: {str(e)}")
    exit(1)

# Use embedding model
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/e5-small-v2",
    model_kwargs={'device': 'cpu'}
)
print("Embedding model loaded successfully")

# Create a directory for documents if it doesn't exist
os.makedirs("documents", exist_ok=True)

# Load and process documents
# Supports PDF, text, and CSV files
print("Loading documents...")
pdf_loader = DirectoryLoader("./documents", glob="**/*.pdf", loader_cls=PyPDFLoader)
text_loader = DirectoryLoader("./documents", glob="**/*.txt", loader_cls=TextLoader)
csv_loader = DirectoryLoader("./documents", glob="**/*.csv", loader_cls=CSVLoader)

pdf_documents = pdf_loader.load() if os.path.exists("./documents") and any(f.endswith('.pdf') for f in os.listdir("./documents")) else []
text_documents = text_loader.load() if os.path.exists("./documents") and any(f.endswith('.txt') for f in os.listdir("./documents")) else []
csv_documents = csv_loader.load() if os.path.exists("./documents") and any(f.endswith('.csv') for f in os.listdir("./documents")) else []

documents = pdf_documents + text_documents + csv_documents
print(f"Loaded {len(pdf_documents)} PDF documents, {len(text_documents)} text documents, and {len(csv_documents)} CSV documents")

if not documents:
    print("No documents found in the 'documents' directory. Creating sample documents for testing.")
    
    # Add a sample text document
    with open("./documents/sample.txt", "w") as f:
        f.write("""
        This is a sample document for testing the RAG system.
        The Telegram bot is built using Python and deployed on Google Cloud Functions.
        It uses Gemini 2.0 Flash Lite for generating responses and e5-small-v2 embeddings for retrieval.
        LangChain is used to connect all components together.
        This bot can answer questions about artificial intelligence, machine learning, and natural language processing.
        Retrieval-Augmented Generation (RAG) combines retrieval of documents with text generation.
        """)
    
    # Add a sample CSV document
    with open("./documents/sample.csv", "w") as f:
        f.write("""topic,description
Artificial Intelligence,AI is the simulation of human intelligence processes by machines.
Machine Learning,ML is a subset of AI that enables systems to learn and improve from experience.
Natural Language Processing,NLP is a field of AI that gives computers the ability to understand text and spoken words.
Retrieval-Augmented Generation,RAG is a technique that combines retrieval of documents with text generation.
Embeddings,Embeddings are vector representations of text that capture semantic meaning.
Vector Database,A vector database is optimized for storing and searching vector embeddings.
""")
    
    # Reload documents
    text_loader = DirectoryLoader("./documents", glob="**/*.txt", loader_cls=TextLoader)
    csv_loader = DirectoryLoader("./documents", glob="**/*.csv", loader_cls=CSVLoader)
    text_documents = text_loader.load()
    csv_documents = csv_loader.load()
    documents = text_documents + csv_documents
    print(f"Created sample documents. Loaded {len(text_documents)} text documents and {len(csv_documents)} CSV documents")

# Split documents into chunks
print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

print(f"Processing {len(texts)} text chunks...")

# Create vector store in Pinecone
try:
    print(f"Uploading to Pinecone index...")
    
    # Process and upload in batches
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...")
        
        # Get embeddings for the batch
        ids = [f"doc_{i + j}" for j in range(len(batch))]
        texts_to_embed = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]
        
        # Get embeddings
        embeddings_batch = embeddings.embed_documents(texts_to_embed)
        
        # Create records
        vectors = []
        for j, (text, metadata, embedding) in enumerate(zip(texts_to_embed, metadatas, embeddings_batch)):
            vectors.append({
                "id": ids[j],
                "values": embedding,
                "metadata": {"text": text, **metadata}
            })
        
        # Upsert to Pinecone
        index.upsert(vectors=vectors)
        print(f"Uploaded batch {i//batch_size + 1}")
    
    print(f"Successfully processed {len(texts)} text chunks into Pinecone")
    print("Your documents are now ready to be queried by the Telegram bot!")
    
    # Print instructions for updating the Cloud Function
    print("\nIMPORTANT: Update your main.py with the following code:")
    print("```python")
    print("# Initialize Pinecone")
    print("from pinecone import Pinecone")
    print("pc = Pinecone(api_key=PINECONE_API_KEY)")
    print(f"index = pc.Index('{INDEX_NAME}')")
    print("```")
    
    print("\nNOTE: This script uses the intfloat/e5-small-v2 embedding model which produces")
    print("384-dimensional vectors. Make sure your Pinecone index is configured with dimension=384.")
    print("If you previously used a different model (like BAAI/bge-base-en-v1.5 with 768 dimensions),")
    print("you'll need to create a new index with the correct dimensions.")
    
except Exception as e:
    print(f"Error uploading documents to Pinecone: {str(e)}")
    print("Please check your Pinecone account and try again.")
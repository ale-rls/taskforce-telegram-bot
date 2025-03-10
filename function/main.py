from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import functions_framework
import os
import json
import logging
import asyncio
import google.generativeai as genai
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.retrieval_qa.base import RetrievalQA

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Get environment variables
BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
INDEX_NAME = os.environ.get('PINECONE_INDEX', 'telegram-bot-docs')

if not BOT_TOKEN:
    logger.error("No TELEGRAM_BOT_TOKEN environment variable found")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash-lite')

# Global variable for RAG chain
rag_chain = None

# Initialize RAG components
def initialize_rag():
    try:
        # Initialize Pinecone with the new API
        logger.info("Initializing Pinecone")
        debug_messages = ["Starting RAG initialization..."]
        
        if not PINECONE_API_KEY:
            error_msg = "Missing Pinecone API key"
            logger.error(error_msg)
            debug_messages.append(f"ERROR: {error_msg}")
            return None, debug_messages
            
        try:
            # No need to initialize Pinecone separately with the new API
            # The PineconeVectorStore will handle this
            debug_messages.append(f"Using Pinecone index: {INDEX_NAME}")
            logger.info(f"Using Pinecone index: {INDEX_NAME}")
        except Exception as e:
            error_msg = f"Failed to connect to Pinecone: {str(e)}"
            logger.error(error_msg)
            debug_messages.append(f"ERROR: {error_msg}")
            return None, debug_messages
        
        # Initialize embeddings
        logger.info("Loading embedding model...")
        debug_messages.append("Loading embedding model...")
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="intfloat/e5-small-v2",
                model_kwargs={'device': 'cpu'}
            )
            logger.info("Embedding model loaded successfully")
            debug_messages.append("Embedding model loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load embedding model: {str(e)}"
            logger.error(error_msg)
            debug_messages.append(f"ERROR: {error_msg}")
            return None, debug_messages
        
        # Connect to existing index using LangChain
        logger.info("Creating LangChain vectorstore...")
        debug_messages.append("Creating LangChain vectorstore...")
        try:
            # Create the vectorstore using the approach from the documentation
            os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY  # Set environment variable for Pinecone
            
            # First, create a Pinecone index object
            pc = Pinecone(api_key=PINECONE_API_KEY)
            index = pc.Index(INDEX_NAME)
            
            # Then create the vectorstore with the index object
            text_field = "text"
            vectorstore = PineconeVectorStore(
                index, embeddings, text_field
            )
            
            logger.info("Vectorstore created successfully")
            debug_messages.append("Vectorstore created successfully")
        except Exception as e:
            error_msg = f"Failed to create vectorstore: {str(e)}"
            logger.error(error_msg)
            debug_messages.append(f"ERROR: {error_msg}")
            return None, debug_messages
        
        # Create retriever
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            logger.info("Retriever created successfully")
            debug_messages.append("Retriever created successfully")
        except Exception as e:
            error_msg = f"Failed to create retriever: {str(e)}"
            logger.error(error_msg)
            debug_messages.append(f"ERROR: {error_msg}")
            return None, debug_messages
        
        # Create LLM
        if not GEMINI_API_KEY:
            error_msg = "Missing Gemini API key"
            logger.error(error_msg)
            debug_messages.append(f"ERROR: {error_msg}")
            return None, debug_messages
            
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", google_api_key=GEMINI_API_KEY)
            logger.info("LLM initialized successfully")
            debug_messages.append("LLM initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize LLM: {str(e)}"
            logger.error(error_msg)
            debug_messages.append(f"ERROR: {error_msg}")
            return None, debug_messages
        
        # Create RetrievalQA chain
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            
            logger.info("RetrievalQA chain created successfully")
            debug_messages.append("RetrievalQA chain created successfully")
        except Exception as e:
            error_msg = f"Failed to create RetrievalQA chain: {str(e)}"
            logger.error(error_msg)
            debug_messages.append(f"ERROR: {error_msg}")
            return None, debug_messages
        
        # Create a simple function to get responses
        def get_response(query):
            logger.info(f"Processing query: {query}")
            try:
                # Use the RetrievalQA chain
                result = qa_chain.invoke({"query": query})
                
                # Extract answer and sources
                answer = result.get("result", "")
                source_documents = result.get("source_documents", [])
                
                # Format sources
                sources = []
                if source_documents:
                    for i, doc in enumerate(source_documents):
                        metadata = doc.metadata
                        source = metadata.get("source", f"Source {i+1}")
                        sources.append(source)
                
                # Return formatted response
                return {
                    "answer": answer,
                    "sources": ", ".join(sources) if sources else "No sources found"
                }
            except Exception as e:
                logger.error(f"Error in get_response: {str(e)}")
                return f"Error processing query: {str(e)}"
            
        logger.info("RAG system initialized successfully")
        debug_messages.append("RAG system initialized successfully")
        return get_response, debug_messages
    except Exception as e:
        error_msg = f"Error initializing RAG: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        return None, [error_msg, "Check logs for detailed traceback"]

# We don't initialize RAG at startup anymore
# rag_chain = initialize_rag()

# Define command handlers
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user_id = update.effective_user.id
    logger.info(f"User {user_id} started the bot")
    
    # Send welcome message with debug info
    welcome_message = (
        "Hello! I'm your RAG-powered bot. Ask me anything about your documents!\n\n"
        "DEBUG INFO:\n"
        f"- Bot Token Available: {BOT_TOKEN is not None}\n"
        f"- Gemini API Key Available: {GEMINI_API_KEY is not None}\n"
        f"- Pinecone API Key Available: {PINECONE_API_KEY is not None}\n"
        f"- Pinecone Index: {INDEX_NAME}\n"
        f"- RAG System Initialized: {rag_chain is not None}\n\n"
        "You can send any message to test the bot."
    )
    
    await update.message.reply_text(welcome_message)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process the user message with RAG system."""
    global rag_chain
    
    user_id = update.effective_user.id
    user_message = update.message.text
    
    await update.message.reply_text(f"Received message: {user_message}")
    logger.info(f"Received message from {user_id}: {user_message}")
    
    # Send typing action
    await update.message.chat.send_action(action="typing")
    
    # Initialize RAG if not already done
    if rag_chain is None:
        await update.message.reply_text("RAG not initialized yet, initializing now...")
        logger.info("RAG not initialized yet, initializing now...")
        rag_chain, debug_messages = initialize_rag()
        
        # Send all debug messages
        await update.message.reply_text("--- RAG Initialization Debug Info ---")
        for msg in debug_messages:
            await update.message.reply_text(msg)
        
        if rag_chain:
            await update.message.reply_text("RAG system initialized successfully!")
        else:
            await update.message.reply_text("Failed to initialize RAG system. Will try to answer without it.")
    
    try:
        if rag_chain:
            # Use RAG system
            await update.message.reply_text("Using RAG system to process your query...")
            try:
                # Call the function (now synchronous)
                response = rag_chain(user_message)
                
                # Handle different response formats
                if isinstance(response, str):
                    await update.message.reply_text(f"RAG Response: {response}")
                elif isinstance(response, dict) and 'result' in response:
                    await update.message.reply_text(f"RAG Response: {response['result']}")
                elif isinstance(response, dict) and 'answer' in response:
                    answer = response['answer']
                    sources = response.get('sources', 'No sources provided')
                    await update.message.reply_text(f"RAG Response: {answer}")
                    await update.message.reply_text(f"Sources: {sources}")
                else:
                    await update.message.reply_text(f"RAG Response: {str(response)}")
            except Exception as inner_e:
                logger.error(f"Error in RAG processing: {str(inner_e)}")
                await update.message.reply_text(f"Error with RAG processing: {str(inner_e)}")
                # Fall back to direct Gemini
                await update.message.reply_text("Falling back to direct Gemini model...")
                response = model.generate_content(user_message)
                await update.message.reply_text(f"Gemini Response: {response.text}")
        else:
            # Fallback to direct Gemini
            await update.message.reply_text("Using direct Gemini model (no RAG)...")
            response = model.generate_content(user_message)
            await update.message.reply_text(f"Gemini Response: {response.text}")
    except Exception as e:
        error_message = f"Error getting response: {str(e)}"
        logger.error(error_message)
        await update.message.reply_text(f"DEBUG ERROR: {error_message}")
        await update.message.reply_text("Sorry, I couldn't process your request at the moment.")

@functions_framework.http
def telegram_bot(request):
    """HTTP Cloud Function to handle Telegram updates."""
    logger.info("Function triggered")
    
    # Only process POST requests
    if request.method != "POST":
        logger.info("Received non-POST request")
        return "Please send a POST request"
    
    try:
        # Parse update
        try:
            update_data = json.loads(request.get_data().decode('utf-8'))
            logger.info(f"Received update: {update_data}")
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse JSON: {str(e)}"
            logger.error(error_msg)
            return f"Invalid JSON: {str(e)}", 400
        
        # Check if BOT_TOKEN is available
        if not BOT_TOKEN:
            error_msg = "Missing BOT_TOKEN environment variable"
            logger.error(error_msg)
            return "Server configuration error: Missing BOT_TOKEN", 500
        
        # Process update asynchronously
        async def process_update():
            try:
                # Create application
                app = Application.builder().token(BOT_TOKEN).build()
                
                # Add handlers
                app.add_handler(CommandHandler("start", start_command))
                app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
                
                # Process update
                await app.initialize()
                update_obj = Update.de_json(update_data, app.bot)
                if update_obj is None:
                    logger.error("Failed to parse Telegram update")
                    return
                
                # Log the type of update we're processing
                if update_obj.message:
                    if update_obj.message.text:
                        logger.info(f"Processing text message: {update_obj.message.text[:50]}...")
                    else:
                        logger.info("Processing non-text message")
                elif update_obj.callback_query:
                    logger.info("Processing callback query")
                else:
                    logger.info(f"Processing other update type: {update_obj}")
                
                await app.process_update(update_obj)
            except Exception as e:
                error_msg = f"Error in process_update: {str(e)}"
                logger.error(error_msg)
                import traceback
                logger.error(traceback.format_exc())
        
        # Run the async function
        asyncio.run(process_update())
        logger.info("Update processed successfully")
        return "OK"
    
    except Exception as e:
        error_msg = f"Error processing update: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        return f"Error: {str(e)}", 500 
# Taskforce Telegram Bot

A Retrieval-Augmented Generation (RAG) powered Telegram bot built with Python, LangChain, Pinecone, and Google's Gemini 2.0 Flash Lite. The bot can answer questions based on documents stored in a Pinecone vector database.

## Features

- **Document Processing**: Supports PDF, TXT, and CSV files
- **RAG System**: Combines document retrieval with generative AI
- **Pinecone Integration**: Vector database for efficient document storage and retrieval
- **Gemini 2.0 Flash Lite**: Google's lightweight generative AI model
- **Cloud Function**: Deployable on Google Cloud Functions

## Setup

### Prerequisites

1. Python 3.9+
2. Telegram Bot Token
3. Google Gemini API Key
4. Pinecone API Key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/taskforce-telegram-bot.git
   cd taskforce-telegram-bot
   ```

2. Install dependencies:
   ```bash
   pip install -r function/requirements.txt
   ```

3. Set up environment variables:
   ```bash
   export TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
   export GEMINI_API_KEY="your_gemini_api_key"
   export PINECONE_API_KEY="your_pinecone_api_key"
   ```

### Document Processing

1. Place your documents in the `documents` folder (supports PDF, TXT, and CSV)
2. Run the document processor:
   ```bash
   python document_processor.py
   ```

   This will:
   - Process and split documents into chunks
   - Generate embeddings using the `intfloat/e5-small-v2` model
   - Upload documents to Pinecone

### Deployment

1. Deploy to Google Cloud Functions:
   ```bash
   gcloud functions deploy telegram-bot \
       --runtime python310 \
       --trigger-http \
       --allow-unauthenticated \
       --entry-point telegram_bot
   ```

2. Set your Telegram bot's webhook:
   ```bash
   curl -F "url=https://your-cloud-function-url" https://api.telegram.org/bot<YOUR_BOT_TOKEN>/setWebhook
   ```

## Usage

1. Start the bot with `/start`
2. Ask questions about your documents
3. The bot will respond with answers and sources

## Configuration

### Environment Variables

| Variable            | Description                          |
|---------------------|--------------------------------------|
| TELEGRAM_BOT_TOKEN  | Your Telegram bot token              |
| GEMINI_API_KEY      | Google Gemini API key                |
| PINECONE_API_KEY    | Pinecone API key                     |
| PINECONE_INDEX      | Pinecone index name (default: telegram-bot-docs) |

### Pinecone Index

- Ensure your Pinecone index is configured with `dimension=384` for the `intfloat/e5-small-v2` embedding model

## File Structure 
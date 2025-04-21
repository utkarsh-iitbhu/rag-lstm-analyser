# RAG-LSTM-Analyser

## B2B Review Insights Platform

A powerful tool that combines LSTM-based sentiment analysis with Retrieval Augmented Generation (RAG) to transform customer reviews into actionable business intelligence.

📋 Table of Contents

- [What is This?](#what-is-this)
- [Why Use It?](#why-use-it)
- [How It Works](#how-it-works)
- [Architecture Diagram](#architecture-diagram)
- [Setup and Installation](#setup-and-installation)
  - [Prerequisites](#prerequisites)
  - [Docker Setup](#docker-setup)
  - [Local Setup](#local-setup)
- [Environment Variables](#environment-variables)
- [Using the Application](#using-the-application)
- [Technical Deep Dive](#technical-deep-dive)
- [Business Applications](#business-applications)
- [Limitations and Future Work](#limitations-and-future-work)

## What is This?

The RAG-LSTM-Analyser is a B2B platform designed to help businesses extract meaningful insights from customer reviews. It leverages two powerful AI techniques:

1. **LSTM Neural Networks**: Deep learning models for accurate sentiment analysis (positive, neutral, negative)
2. **Retrieval Augmented Generation (RAG)**: Context-aware response generation using historical review data

This dual approach provides both quick sentiment classification and in-depth business insights that can drive strategic decisions.

## Why Use It?

B2B companies often struggle to extract actionable intelligence from large volumes of customer feedback. This platform solves that problem by:

- **Automating sentiment analysis** across thousands of reviews
- **Identifying recurring themes and patterns** that might be missed by manual analysis
- **Highlighting specific pain points** that need immediate attention
- **Providing context from similar historical reviews** for better understanding
- **Generating strategic recommendations** based on customer feedback
- **Supporting multilingual reviews** through automatic translation

## How It Works

The application follows a straightforward process flow:

1. **Input Collection**: Users enter a customer review and an optional business question.
2. **Language Detection & Translation**: Non-English reviews are automatically translated.
3. **Sentiment Analysis**: The LSTM model classifies sentiment as positive (1), neutral (0), or negative (-1).
4. **Similar Review Retrieval**: The RAG system finds similar historical reviews from the database.
5. **Context-Aware Analysis**: The LLM analyzes the current review in context of similar past reviews.
6. **Business Intelligence Generation**: Strategic insights and recommendations are presented.
7. **Result Visualization**: Sentiment scores and business insights are displayed in an intuitive interface.

All of this happens in seconds, giving business users immediate access to powerful analytics.

## Architecture Diagram

```
┌────────────────┐      ┌───────────────────┐      ┌───────────────────┐
│                │      │                   │      │                   │
│  Customer      │─────▶│  Streamlit Web    │─────▶│  Language         │
│  Review Input  │      │  Interface        │      │  Detection        │
│                │      │                   │      │                   │
└────────────────┘      └───────────────────┘      └─────────┬─────────┘
                                                             │
┌────────────────┐      ┌───────────────────┐      ┌─────────▼─────────┐
│                │      │                   │      │                   │
│  Business      │◀────▶│  Result           │◀─────│  Translation      │
│  Insights      │      │  Visualization    │      │  (if needed)      │
│                │      │                   │      │                   │
└────────────────┘      └───────────────────┘      └─────────┬─────────┘
       ▲                                                      │
       │                                                      ▼
┌──────┴───────┐      ┌───────────────────┐      ┌───────────────────┐
│              │      │                   │      │                   │
│  OpenAI LLM  │◀────▶│  RAG System       │◀─────│  LSTM Sentiment   │
│  (GPT-4)     │      │  (Pinecone)       │      │  Analysis         │
│              │      │                   │      │                   │
└──────────────┘      └───────────────────┘      └───────────────────┘
```

The architecture follows a logical flow:

1. User inputs a review through the Streamlit interface
2. The system detects the language and translates if necessary
3. The LSTM model analyzes sentiment
4. The RAG system retrieves similar historical reviews from Pinecone
5. OpenAI's LLM generates business insights based on all available context
6. Results are visualized in an interactive interface

## Setup and Installation

### Prerequisites

- Docker installed on your system
- Model weights and data (download link provided separately)
- OpenAI API key
- Pinecone API key and environment

### Docker Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/utkarsh-iitbhu/rag-lstm-analyser.git
   cd rag-lstm-analyser
   ```
2. Download the model weights and data:

   - Download from the provided link
   - Extract and place in `weights/` and `data/` folders in your project directory
3. Create an `.env` file with your API keys:

   ```
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_API_ENV=your_pinecone_environment
   INDEXNAME=your_pinecone_index_name
   ```
4. Build and run the Docker container:

   ```bash
   # Build the Docker image
   docker build -t rag-lstm-analyser .

   # Run the container with relative paths to your weights and data
   docker run -p 8501:8501 \
     --env-file .env \
     -v $(pwd)/weights:/app/weights \
     -v $(pwd)/data:/app/data \
     rag-lstm-analyser
   ```

   Note: This command uses the current directory's `weights/` and `data/` folders. Make sure these exist before running.
5. Access the application in your browser:

   ```
   http://localhost:8501
   ```

### Using Pre-built Docker Image

You can also run this application by pulling the pre-built Docker image from Docker Hub:

```bash
# Pull the image from Docker Hub
docker pull lordsahu/rag-lstm-analyser:latest

# Run the container using your local .env file and data/weights directories
docker run -p 8501:8501 \
  --env-file .env \
  -v $(pwd)/weights:/app/weights \
  -v $(pwd)/data:/app/data \
  lordsahu/rag-lstm-analyser:latest
```

This is the fastest way to get started if you don't need to modify the application code.

### Local Setup

If you prefer not to use Docker:

1. Clone the repository:

   ```bash
   git clone https://github.com/utkarsh-iitbhu/rag-lstm-analyser.git
   cd rag-lstm-analyser
   ```
2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Download and place model weights and data in the appropriate directories.
4. Create an `.env` file with your API keys as described above.
5. Run the application:

   ```bash
   streamlit run app.py
   ```

## Environment Variables

The application requires the following environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key
- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_API_ENV`: Your Pinecone environment (e.g., "us-west1-gcp")
- `INDEXNAME`: The name of your Pinecone index

## Using the Application

1. **Enter a Customer Review**: Type or paste a customer review in the text area.
2. **Enter a Business Question**: Specify what business insights you're looking for.
3. **Click "Analyze Review"**: The system will process the review and generate insights.
4. **View Results**: See sentiment analysis and business intelligence insights in the tabbed interface.
5. **Explore History**: Review past analyses in the history tab.

## Technical Deep Dive

### Sentiment Analysis with LSTM

The application uses a Long Short-Term Memory (LSTM) neural network for sentiment analysis. This deep learning model:

- Understands sequential text data and contextual relationships
- Processes text through word embeddings
- Classifies sentiment into three categories: positive (1), neutral (0), and negative (-1)
- Provides confidence scores for each classification

The model was trained on a large dataset of product reviews and fine-tuned for accuracy across different product categories.

### Retrieval Augmented Generation (RAG)

The RAG system enhances LLM responses with relevant context from a database of historical reviews:

1. **Embedding Generation**: Reviews are converted to vector embeddings
2. **Vector Search**: Pinecone finds semantically similar reviews
3. **Context Integration**: Similar reviews are provided as context to the LLM
4. **Response Generation**: The LLM generates insights based on both the current review and historical context

This approach eliminates hallucinations and grounds the AI's responses in real customer feedback.

## Business Applications

This platform is particularly valuable for:

- **Product Managers**: Identify product issues and prioritize improvements
- **Customer Experience Teams**: Understand pain points and improve customer journey
- **Marketing Departments**: Extract messaging that resonates with customers
- **Executive Leadership**: Get data-driven insights for strategic decisions
- **Competitive Analysis**: Compare product perception against competitors

## Limitations and Future Work

Current limitations:

- Requires API keys for OpenAI and Pinecone
- LSTM model may need periodic retraining as language evolves
- Processing very long reviews may require chunking

Future development plans:

- Add support for batch processing multiple reviews
- Implement custom fine-tuned models for specific industries
- Add visualization dashboards for trend analysis
- Integrate with CRM and product management systems

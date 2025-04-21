"""
B2B Review Insights Platform
----------------------------
This application processes customer reviews using sentiment analysis and RAG (Retrieval-Augmented Generation)
to provide actionable business intelligence insights.
"""

import streamlit as st
import numpy as np
import os
import time
import pickle
from dotenv import load_dotenv
from langdetect import detect
from googletrans import Translator

# ML/AI imports
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pinecone
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_pinecone import Pinecone
from langchain.embeddings import OpenAIEmbeddings


# Configuration Constants
MAX_LENGTH = 100
SENTIMENT_MAP = {
    0: {"label": "Negative", "value": -1, "color": "🔴", "emoji": "😠"},
    1: {"label": "Neutral", "value": 0, "color": "🟡", "emoji": "😐"},
    2: {"label": "Positive", "value": 1, "color": "🟢", "emoji": "😊"}
}


def load_sentiment_resources():
    """
    Load the trained sentiment analysis model and tokenizer.
    
    Returns:
        tuple: (model, tokenizer) if successful, (None, None) otherwise
    """
    try:
        model = load_model('weights/lstm_model.h5')
        with open('weights/tokenizer.pkl', 'rb') as file:
            tokenizer = pickle.load(file)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading sentiment model or tokenizer: {e}")
        return None, None


def initialize_rag():
    """
    Initialize Retrieval-Augmented Generation components using Pinecone and OpenAI.
    
    Returns:
        RetrievalQA: Initialized QA chain or None if initialization fails
    """
    try:
        # Initialize LLM
        llm = ChatOpenAI(openai_api_key=KEY, model_name='gpt-4o-mini')
        
        # Initialize embeddings
        embedding = OpenAIEmbeddings(openai_api_key=KEY)
        
        # Connect to Pinecone
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
        index = pc.Index(INDEXNAME)
        
        # Initialize Langchain Pinecone wrapper
        docsearch = Pinecone(index=index, embedding=embedding, text_key="text")
        
        # Enhanced prompt template for business insights
        prompt_template = """
        You are a B2B business intelligence expert analyzing product reviews for actionable insights.

        First, determine the sentiment of the following customer review: "{question}"
        
        Then, using the historical review data provided in the context, provide a comprehensive business analysis focusing on:

        1. Sentiment validation: Does your sentiment analysis agree with the LSTM model's prediction? If not, explain why.
        
        2. Key business themes and patterns: Identify recurring themes across similar reviews.
        
        3. Customer pain points: What specific issues are customers experiencing with this product category?
        
        4. Competitive analysis: How does this product compare to alternatives mentioned in the reviews?
        
        5. Business impact: Quantify potential revenue/customer satisfaction impact of these issues.
        
        6. Strategic recommendations: Provide 3-5 actionable, prioritized recommendations for business improvement.

        Context (historical similar reviews): {context}
        
        Format your response with clear headings and bullet points for easy executive review. Cite specific examples from the reviews to support your analysis.
        """
        
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain_type_kwargs = {"prompt": PROMPT}
        
        # Initialize QA chain
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=docsearch.as_retriever(search_kwargs={'k': 10}),
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs,
        )
        
        return qa
    except Exception as e:
        st.error(f"Error initializing RAG components: {e}")
        return None


def detect_and_translate(review_text):
    """
    Detect language of text and translate to English if needed.
    
    Args:
        review_text (str): The text to analyze and potentially translate
        
    Returns:
        tuple: (translated_text, language_code)
    """
    try:
        language = detect(review_text)
        if language != 'en':
            translation = translator.translate(review_text, src=language, dest='en').text
            return translation, language
        return review_text, language
    except Exception as e:
        st.error(f"Error during translation: {e}")
        return review_text, 'en'


def predict_sentiment(review_text):
    """
    Predict sentiment of a given text using the loaded model.
    
    Args:
        review_text (str): Text to analyze sentiment
        
    Returns:
        tuple: Sentiment details (label, confidence, class, all probabilities, color, emoji)
    """
    sequences = tokenizer.texts_to_sequences([review_text])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LENGTH)
    prediction = sentiment_model.predict(padded_sequences)
    
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]
    
    sentiment_label = SENTIMENT_MAP[predicted_class]["label"]
    sentiment_value = SENTIMENT_MAP[predicted_class]["value"]
    sentiment_color = SENTIMENT_MAP[predicted_class]["color"]
    sentiment_emoji = SENTIMENT_MAP[predicted_class]["emoji"]
    
    return sentiment_label, confidence, predicted_class, prediction[0], sentiment_color, sentiment_emoji


def get_business_insights(review_text, business_question):
    """
    Generate business insights from review text using RAG system.
    
    Args:
        review_text (str): The review text to analyze
        business_question (str): The business question to answer
        
    Returns:
        tuple: (insights_text, source_documents)
    """
    if qa_chain is None:
        return "RAG system not initialized properly. Please check your API keys and environment variables.", []
    
    try:
        query = review_text
        result = qa_chain.invoke({'query': query})
        top_sources = result.get('source_documents', [])[:5]
        return result['result'], top_sources
    except Exception as e:
        return f"Error generating business insights: {str(e)}", []


def add_to_history(review_text, sentiment_label, confidence, predicted_class, all_probs, 
                  business_question=None, business_insights=None, sources=None,
                  language=None, translated_text=None, sentiment_color=None, sentiment_emoji=None):
    """
    Add analysis results to session history.
    
    Args:
        Multiple parameters representing analysis results
    """
    st.session_state.history.append((
        review_text, sentiment_label, confidence, predicted_class, all_probs,
        business_question, business_insights, sources, language, translated_text,
        sentiment_color, sentiment_emoji
    ))


def render_sidebar():
    """Render the application sidebar with information and instructions."""
    with st.sidebar:
        st.title("🏠 B2B Review Insights")
        
        st.markdown("---")
        st.subheader("About This Platform")
        st.write("""
        This platform combines advanced AI technologies to transform customer reviews into actionable business intelligence:
        
        🧠 **LSTM Neural Network**: Analyzes sentiment with deep learning
        
        🔍 **RAG Technology**: Leverages historical review data for context-aware insights
        
        💼 **Business Intelligence**: Converts raw data into strategic recommendations
        """)
        
        st.markdown("---")
        st.subheader("How It Works")
        st.write("""
        1. Enter a customer review
        2. Ask a business question
        3. Get instant sentiment analysis
        4. Receive strategic business insights
        5. Make data-driven decisions
        """)
        
        st.markdown("---")
        st.caption("© 2025 B2B Review Insights | Powered by AI")


def render_analysis_tab():
    """Render the review analysis input tab."""
    st.header('Enter Customer Review', anchor='enter-review')
    sample_reviews = [
        "Select a sample review or write your own below:",
        "I researched about tablets and based on my interests I choose the Amazon Fire 7 tablet and protector, I wanted reading and games mostly.",
        "I was looking for a kindle whitepaper. I saw online for $80. What a deal. I ordered it on line and picked it up in the store. I got it home and couldn't adjust the brightness. After a lengthy time with online customer service I called customer service. After 20 minuets with speaking to a female Elmer Fud that doesn't speak english well I figured I would just return it. Although it looks Identical to the $120 model, you can not adjust the brightness. That would have been good information before I bought it.",
        "The Amazon echo is excellent. Easy to set up, and we now use it to automate the entire house from our air-conditioning thermostat to every light in the house with the Phillips hue bulbs. We went and got one for everyone in our family.",
        "This kindle does its job but I would buy one which the screen is brighter. There are times that it's difficult to read because screen is not too bright",
        "I think it is worth the sale price of $150.00 but it has limitations. For example it shows my calendar but you can only see the first item, then if you ask for more, you see the next 3 or so. I The screen is updating with trivia...some interesting some not so much. I bet the next generation is a lot better",
    ]
    selected_sample = st.selectbox("", sample_reviews)

    review_text = selected_sample if selected_sample != sample_reviews[0] else ""
    review_text = st.text_area("Customer Review:", value=review_text, height=150)

    st.header('Business Question (for RAG Analysis)', anchor='business-question')
    default_question = "What are the main pain points in our products and how can we improve?"
    business_question = st.text_area("Enter a business question about your products:", 
                                  value=default_question,
                                  height=80)

    if st.button('Analyze Review', type="primary", use_container_width=True):
        process_review_analysis(review_text, business_question)


def process_review_analysis(review_text, business_question):
    """
    Process the review analysis when the button is clicked.
    
    Args:
        review_text (str): The review text to analyze
        business_question (str): The business question to answer
    """
    if not review_text:
        st.warning("Please enter a review text.")
        return
        
    with st.spinner('Analyzing review...'):
        progress_bar = st.progress(0)
        
        # Step 1: Translation if needed
        progress_bar.progress(20)
        translated_text, language = detect_and_translate(review_text)
        
        # Step 2: Sentiment analysis
        progress_bar.progress(40)
        sentiment_label, confidence, predicted_class, all_probs, sentiment_color, sentiment_emoji = predict_sentiment(translated_text)
        
        # Step 3: Business insights
        progress_bar.progress(60)
        st.info("Retrieving relevant historical reviews...")
        time.sleep(0.5)
        progress_bar.progress(80)
        business_insights, sources = get_business_insights(translated_text, business_question)
        
        progress_bar.progress(100)
        time.sleep(0.5)
        progress_bar.empty()
        
        st.success("Analysis complete!")
        display_analysis_results(
            review_text, translated_text, language, 
            sentiment_label, confidence, predicted_class, all_probs, sentiment_emoji,
            business_question, business_insights, sources
        )
        
        # Add to history
        add_to_history(
            review_text, sentiment_label, confidence, predicted_class, all_probs,
            business_question, business_insights, sources, language, translated_text,
            sentiment_color, sentiment_emoji
        )


def display_analysis_results(review_text, translated_text, language, 
                           sentiment_label, confidence, predicted_class, all_probs, sentiment_emoji,
                           business_question, business_insights, sources):
    """
    Display the analysis results in a structured format.
    
    Args:
        Multiple parameters representing analysis results to display
    """
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Sentiment Analysis")
        
        sentiment_container = st.container(border=True)
        with sentiment_container:
            st.markdown(f"### {sentiment_emoji} {sentiment_label}")
            
            confidence_percentage = int(confidence * 100)
            
            if predicted_class == 0:
                st.markdown(f"<h2 style='color: #ff4b4b; text-align: center;'>{confidence_percentage}%</h2>", unsafe_allow_html=True)
            elif predicted_class == 1:
                st.markdown(f"<h2 style='color: #ffaa00; text-align: center;'>{confidence_percentage}%</h2>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h2 style='color: #00cc00; text-align: center;'>{confidence_percentage}%</h2>", unsafe_allow_html=True)
        
        with st.expander("Detailed Sentiment Breakdown"):
            for i, prob in enumerate(all_probs):
                st.write(f"{SENTIMENT_MAP[i]['color']} {SENTIMENT_MAP[i]['label']} ({SENTIMENT_MAP[i]['value']}): {int(prob * 100)}%")
        
        if language != 'en':
            with st.expander("Translation Information"):
                st.info(f"Original Language: {language}")
                st.write(f"Translated Review: {translated_text}")
    
    with col2:
        st.subheader("Business Intelligence Insights")
        
        insights_container = st.container(border=True)
        with insights_container:
            st.markdown(business_insights)
        
        with st.expander("View Source Reviews (Top 5)"):
            for i, doc in enumerate(sources):
                src_container = st.container(border=True)
                with src_container:
                    st.markdown(f"**Source {i+1}:**")
                    st.write(doc.page_content)


def render_history_tab():
    """Render the analysis history tab."""
    st.header("Analysis History")
    
    if not st.session_state.history:
        st.info("No analysis history available. Analyze a review to see the history.")
        return
        
    for idx, (review, sentiment, conf, pred_class, probs, bq, insights, sources, lang, translated, sent_color, sent_emoji) in enumerate(reversed(st.session_state.history)):
        with st.container(border=True):
            st.subheader(f"Analysis #{len(st.session_state.history) - idx}")
            
            st.markdown("**Review:**")
            st.markdown(f"> {review}")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                sentiment_container = st.container(border=True)
                with sentiment_container:
                    st.markdown(f"### {sent_emoji} {sentiment}")
                    
                    confidence_percentage = int(conf * 100)
                    
                    if pred_class == 0:
                        st.markdown(f"<h3 style='color: #ff4b4b; text-align: center;'>{confidence_percentage}%</h3>", unsafe_allow_html=True)
                    elif pred_class == 1:
                        st.markdown(f"<h3 style='color: #ffaa00; text-align: center;'>{confidence_percentage}%</h3>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h3 style='color: #00cc00; text-align: center;'>{confidence_percentage}%</h3>", unsafe_allow_html=True)
            
            with col2:
                if bq:
                    st.markdown(f"**Business Question:** {bq}")
                
                if insights:
                    st.markdown("**Business Insights Summary:**")
                    insights_lines = insights.split('\n')
                    insights_summary = '\n'.join(insights_lines[:3]) + "..."
                    st.markdown(insights_summary)
                    
                    details_button = st.button(f"📋 View Full Analysis #{len(st.session_state.history) - idx}", key=f"details_{idx}")
                    if details_button:
                        st.session_state[f"show_details_{idx}"] = True
            
            if st.session_state.get(f"show_details_{idx}", False):
                with st.expander("Full Analysis", expanded=True):
                    st.markdown("### Complete Business Insights")
                    st.markdown(insights)
                    
                    st.markdown("### Source Reviews")
                    for i, doc in enumerate(sources[:5] if sources else []):
                        st.markdown(f"**Source {i+1}:**")
                        st.write(doc.page_content)
                        st.markdown("---")
                
                if st.button(f"Close Details #{len(st.session_state.history) - idx}", key=f"close_{idx}"):
                    st.session_state[f"show_details_{idx}"] = False


def main():
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="B2B Review Insights Platform",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize history in session state if not present
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    st.title('📊 B2B Review Insights Platform')
    st.markdown("### Transform customer reviews into actionable business intelligence")
    
    # Create tabs for different sections
    tab_input, tab_history = st.tabs(["📝 Review Analysis", "📚 Analysis History"])
    
    with tab_input:
        render_analysis_tab()
    
    with tab_history:
        render_history_tab()


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
    INDEXNAME = os.getenv("INDEXNAME")
    
    # Initialize components
    sentiment_model, tokenizer = load_sentiment_resources()
    qa_chain = initialize_rag()
    translator = Translator()
    
    # Run the app
    main()
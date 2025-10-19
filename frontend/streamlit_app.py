
import streamlit as st
import requests
import json
import time
import pandas as pd
import altair as alt

# API configuration
API_URL = "http://localhost:8000"

# Set page configuration with custom theme
st.set_page_config(
    page_title="Question Answering System",
    page_icon="🧐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling with dark mode
st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Dark mode theme */
    .main {
        background-color: #121212;
        color: #e0e0e0;
    }
    
    .stApp {
        background-color: #121212;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #ffffff;
    }
    
    .subheader {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        color: #e0e0e0;
    }
    
    /* Modern card style */
    .card {
        padding: 1.5rem;
        border-radius: 0.8rem;
        background-color: #1e1e1e;
        margin-bottom: 1.5rem;
        border: 1px solid #333333;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Result card styling */
    .result-card {
        border-left: 4px solid #4CAF50;
    }
    
    .answer-box {
        background-color: #2A2A2A;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    
    .context-box {
        background-color: #2A2A2A;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        max-height: 300px;
        overflow-y: auto;
    }
    
    .metrics-container {
        display: flex;
        gap: 15px;
        margin: 15px 0;
    }
    
    .highlight {
        background-color: rgba(25, 118, 210, 0.15);
        padding: 0.25rem;
        border-radius: 0.25rem;
        color: #64b5f6;
        font-weight: bold;
    }
    
    .team-member {
        padding: 1.5rem;
        border-radius: 0.8rem;
        background-color: #1e1e1e;
        margin-bottom: 1.5rem;
        text-align: center;
        border: 1px solid #333333;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    
    .team-member:hover {
        transform: translateY(-2px);
        border-color: #4CAF50;
    }
    
    .footer {
        text-align: center;
        padding: 1rem;
        font-size: 0.8rem;
        color: #9e9e9e;
        margin-top: 2rem;
        border-top: 1px solid #333;
    }
    
    /* Metric cards with dark theme */
    .metric-card {
        padding: 1.5rem;
        border-radius: 0.8rem;
        background-color: #1e1e1e;
        margin-bottom: 1.5rem;
        text-align: center;
        border: 1px solid #333333;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #4CAF50;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #64b5f6;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #9e9e9e;
        margin-top: 0.5rem;
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 0.5rem;
        font-weight: 500;
        background-color: #1976d2;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(25, 118, 210, 0.4);
        background-color: #1565c0;
    }
    
    /* Input fields styling */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #2d2d2d;
        color: #e0e0e0;
        border: 1px solid #444444;
        border-radius: 0.5rem;
    }
    
    .stTextInput>label, .stTextArea>label {
        color: #e0e0e0;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1lcbmhc {
        background-color: #1a1a1a;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #1a1a1a;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #2d2d2d;
        color: #e0e0e0;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        margin-right: 0.5rem;
        transition: background-color 0.2s, color 0.2s;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #1976d2;
        color: white;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: #4CAF50 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: bold;
        color: #4CAF50 !important;
    }
    
    /* Success/Error message styling */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 8px !important;
    }
    
    /* Remove all white bars and fix spacing */
    hr {
        display: none !important;
    }
    
    .css-18e3th9, .css-1d391kg {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0 !important;
    }
    
    div.css-1r6slb0.e1tzin5v2, div.css-12oz5g7.e1tzin5v2 {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Fix for white horizontal bar under tabs */
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }
    
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: transparent !important;
    }
    
    /* Streamlit dataframe styling */
    .dataframe {
        background-color: #1e1e1e;
        color: #e0e0e0;
        border: 1px solid #333333;
        border-radius: 0.5rem;
    }
    
    .dataframe th {
        background-color: #2d2d2d;
        color: #e0e0e0;
        padding: 0.5rem;
    }
    
    .dataframe td {
        background-color: #1e1e1e;
        color: #e0e0e0;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# API interaction functions

# Functions for API interaction
def get_answer(question, context):
    """
    Get answer from the API with enhanced error handling.
    """
    if not question or not context:
        st.error("❌ Both question and context are required.")
        return None
        
    try:
        with st.spinner("Processing your question..."):
            response = requests.post(
                f"{API_URL}/answer",
                json={"question": question, "context": context},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"❌ API Error: Status code {response.status_code}")
                if response.status_code == 422:
                    st.info("💡 Hint: Check that your question is properly formatted.")
                return None
                
    except requests.exceptions.ConnectionError:
        st.error("❌ Failed to connect to the API. Make sure the backend server is running.")
        st.info("💡 Hint: Run 'uvicorn main:app --reload' in your backend directory.")
        return None
    except requests.exceptions.Timeout:
        st.error("❌ Request timed out. The server might be overloaded or the question too complex.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

def check_api_status():
    """
    Check if the API is running with detailed status information.
    """
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            st.success("✅ API is running and healthy")
            return True
        else:
            st.warning(f"⚠️ API responded with status code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Please check if the backend server is running.")
        st.info("💡 Hint: Run 'uvicorn main:app --reload' in your backend directory.")
        return False
    except Exception as e:
        st.error(f"❌ Error checking API status: {str(e)}")
        return False

# Example data with context and questions
example_data = {
    "Geography - Capital City": {
        "context": "The capital of Nepal is Kathmandu. It is located in the Kathmandu Valley and is the largest city in Nepal. Kathmandu is the political, commercial, cultural, and historical center of Nepal.",
        "question": "What is the capital of Nepal?"
    },
    "Science - Solar System": {
        "context": "The Solar System consists of the Sun and the objects that orbit it, either directly or indirectly. Of the objects that orbit the Sun directly, the largest are the eight planets, with the remainder being smaller objects, such as the five dwarf planets and small Solar System bodies.",
        "question": "How many planets are in the Solar System?"
    },
    "History - World War II": {
        "context": "World War II, also known as the Second World War, was a global war that lasted from 1939 to 1945. It involved the vast majority of the world's countries—including all the great powers—forming two opposing military alliances: the Allies and the Axis. The war began with the invasion of Poland by Nazi Germany on 1 September 1939.",
        "question": "When did World War II begin?"
    },
    "Literature - Shakespeare": {
        "context": "William Shakespeare was an English poet, playwright, and actor, widely regarded as the greatest writer in the English language and the world's greatest dramatist. He is often called England's national poet and the 'Bard of Avon'. His extant works, including collaborations, consist of approximately 39 plays, 154 sonnets, two long narrative poems, and a few other verses.",
        "question": "How many plays did Shakespeare write?"
    }
}

# Example loader function
def load_example(example_key):
    if example_key in example_data:
        st.session_state.context = example_data[example_key]["context"]
        st.session_state.question = example_data[example_key]["question"]
        st.rerun()  # Force a rerun to update the UI with new values

# Initialize session state variables
if 'context' not in st.session_state:
    st.session_state.context = ""
if 'question' not in st.session_state:
    st.session_state.question = ""
if 'result' not in st.session_state:
    st.session_state.result = None

# Sidebar
with st.sidebar:
    # st.image("https://img.icons8.com/color/96/000000/question-mark.png", width=80)
    st.markdown("<h2>Question Answering System</h2>", unsafe_allow_html=True)
    
    
    
    # API status
    api_status = check_api_status()
    st.write("API Status:")
    if api_status:
        st.success("✅ API is running")
    else:
        st.error("❌ API is not running")
        st.info("Start the backend server with: `uvicorn src.serve:app --host 0.0.0.0 --port 8000`")
    
    # Example Questions section
    st.markdown("<h3>📝 Example Questions</h3>", unsafe_allow_html=True)
    
    for example_key, example_content in example_data.items():
        with st.expander(example_key):
            st.markdown(f"**Context:** {example_content['context']}")
            st.markdown(f"**Question:** {example_content['question']}")
            if st.button("Use this example", key=f"btn_{example_key}"):
                load_example(example_key)

# App title and description
st.markdown("<h1 class='main-header'>Question Answering System</h1>", unsafe_allow_html=True)
st.markdown("""
<p>Ask any question about a paragraph and get instant answers powered by NLP</p>
""", unsafe_allow_html=True)

# Create tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["Home", "Model Info", "Team", "Evaluation"])

with tab1:
    # Modern card-style input area
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 class='subheader'>📄 Context</h3>", unsafe_allow_html=True)
    context = st.text_area(
    "Enter a paragraph of text:",
    key="context",
    height=200,
    placeholder="Paste your text here...",
    help="The model will use this text to find answers to your questions."
)
    
    # Input for question
    st.markdown("<h3 class='subheader'>❓ Question</h3>", unsafe_allow_html=True)
    question = st.text_input(
    "Ask a question about the text:",
    key="question",
    placeholder="What, who, when, where, why, how...?",
    help="Your question should be answerable from the context above."
)
    
    # Modern button layout
    col1, col2, col3 = st.columns([1, 1, 5])
    with col1:
        submit = st.button("🔍 Get Answer", type="primary", disabled=not api_status)
    with col2:
        clear = st.button("🗑️ Clear")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Clear inputs
    if clear:
        st.session_state.context = ""
        st.session_state.question = ""
        st.session_state.result = None
        st.rerun()
    
    # Display answer in a modern card layout
    if submit and question and context:
        # Check API status first
        api_status = check_api_status()
        
        if api_status:
            with st.spinner("Finding answer..."):
                start_time = time.time()
                result = get_answer(question, context)
                elapsed_time = time.time() - start_time
                st.session_state.result = result
                
                if result:
                    st.success(f"✅ Found answer in {elapsed_time:.2f} seconds")
                    
                    # Modern Output Display Section
                    st.markdown("<div class='card result-card'>", unsafe_allow_html=True)
                    st.markdown("<h3 class='subheader'>🔍 Answer</h3>", unsafe_allow_html=True)
                    
                    # Check if 'answer' key exists in the result
                    if 'answer' in result:
                        # Display answer with confidence
                        st.markdown(f"<div class='answer-box'><h2>{result['answer']}</h2></div>", unsafe_allow_html=True)
                        
                        # Show confidence score with modern styling
                        confidence = result["score"] * 100
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.progress(min(confidence / 100, 1.0))
                        with col2:
                            st.metric("Confidence", f"{confidence:.1f}%")
                        
                        # Show answer position in context
                        start, end = result["start"], result["end"]
                        highlighted_text = (
                            context[:start] + 
                            f"<span class='highlight'>{context[start:end]}</span>" + 
                            context[end:]
                        )
                        
                        # Modern metrics display
                        st.markdown("<div class='metrics-container'>", unsafe_allow_html=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                            st.markdown(f"<p class='metric-value'>{start}-{end}</p>", unsafe_allow_html=True)
                            st.markdown("<p class='metric-label'>Answer Position</p>", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                            st.markdown(f"<p class='metric-value'>{end-start}</p>", unsafe_allow_html=True)
                            st.markdown("<p class='metric-label'>Answer Length</p>", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Confidence visualization with modern styling
                        st.markdown("<h4>📊 Confidence Visualization</h4>", unsafe_allow_html=True)
                        confidence_data = pd.DataFrame({
                            'Metric': ['Confidence'],
                            'Value': [confidence]
                        })
                        
                        chart = alt.Chart(confidence_data).mark_bar().encode(
                            x=alt.X('Value', scale=alt.Scale(domain=[0, 100])),
                            y='Metric',
                            color=alt.value('#4CAF50')  # Modern green color
                        ).properties(height=100)
                        
                        st.altair_chart(chart, use_container_width=True)
                        
                        # Show answer in context with modern styling
                        with st.expander("📝 View answer in context"):
                            st.markdown("<div class='context-box'>", unsafe_allow_html=True)
                            st.markdown(highlighted_text, unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.error("❌ No answer found in the response")
                        st.json(result)  # Display the actual response for debugging
                    
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("❌ Cannot connect to the backend API. Please make sure it's running.")
            st.info("💡 Try running 'uvicorn main:app --reload' in your backend directory.")
    elif submit:
        st.warning("Please provide both a question and context.")

with tab2:
    st.markdown("<h2 class='subheader'>Dataset & Model Information</h2>", unsafe_allow_html=True)
    
    # Dataset information
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>Dataset</h3>", unsafe_allow_html=True)
    st.markdown("""
    <p>This Question Answering system is trained on the Stanford Question Answering Dataset (SQuAD).</p>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("<p class='metric-value'>SQuAD v1.1</p>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Dataset Version</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("<p class='metric-value'>100,000+</p>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Training Samples</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <p>SQuAD (Stanford Question Answering Dataset) is a reading comprehension dataset consisting of questions posed on a set of Wikipedia articles, 
    where the answer to each question is a segment of text from the corresponding reading passage.</p>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Model information
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>Model</h3>", unsafe_allow_html=True)
    st.markdown("""
    <p>We use a fine-tuned BERT-large-uncased model for question answering.</p>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("<p class='metric-value'>BERT-large</p>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Base Model</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("<p class='metric-value'>3</p>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Training Epochs</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("<p class='metric-value'>5e-5</p>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Learning Rate</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <p>BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based machine learning technique for natural language processing pre-training developed by Google.</p>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Performance metrics
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>Performance Metrics</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("<p class='metric-value'>87.2%</p>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Exact Match (EM) Score</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("<p class='metric-value'>93.1%</p>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>F1 Score</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Create a DataFrame for the metrics
    metrics_data = pd.DataFrame({
        'Metric': ['Exact Match', 'F1 Score'],
        'Value': [87.2, 93.1]
    })
    
    # Create a bar chart
    chart = alt.Chart(metrics_data).mark_bar().encode(
        x=alt.X('Value', scale=alt.Scale(domain=[0, 100])),
        y='Metric',
        color=alt.condition(
            alt.datum.Metric == 'Exact Match',
            alt.value('#4287f5'),
            alt.value('#42c5f5')
        )
    ).properties(height=200)
    
    st.altair_chart(chart, use_container_width=True)
    
    st.markdown("""
    <p><strong>Exact Match (EM):</strong> The percentage of predictions that exactly match any of the ground truth answers.</p>
    <p><strong>F1 Score:</strong> The average overlap between the prediction and ground truth answer, treating them as bags of tokens.</p>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<h2 class='subheader'>Team Information</h2>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>Question Answering System using NLP</h3>", unsafe_allow_html=True)
    st.markdown("<p>A collaborative project to build an efficient question answering system using natural language processing techniques.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Team members
    st.markdown("<h3>Team Members</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='team-member'>", unsafe_allow_html=True)
        st.markdown("<h4>Ajit</h4>", unsafe_allow_html=True)
        st.markdown("<p>Data Preprocessing</p>", unsafe_allow_html=True)
        st.markdown("<p>Responsible for cleaning and preparing the SQuAD dataset for training.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='team-member'>", unsafe_allow_html=True)
        st.markdown("<h4>Aavash</h4>", unsafe_allow_html=True)
        st.markdown("<p>Model Fine-tuning</p>", unsafe_allow_html=True)
        st.markdown("<p>Worked on fine-tuning the BERT model for question answering task.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='team-member'>", unsafe_allow_html=True)
        st.markdown("<h4>Austin</h4>", unsafe_allow_html=True)
        st.markdown("<p>Backend Integration</p>", unsafe_allow_html=True)
        st.markdown("<p>Developed the FastAPI backend for serving model predictions.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='team-member'>", unsafe_allow_html=True)
        st.markdown("<h4>Sumesh</h4>", unsafe_allow_html=True)
        st.markdown("<p>UI/UX & Frontend</p>", unsafe_allow_html=True)
        st.markdown("<p>Designed and implemented the user interface using Streamlit.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Project details
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>Project Details</h3>", unsafe_allow_html=True)
    st.markdown("""
    <p>This project was developed as part of the *Eight semester project for the course COMP 473(Specch and Language Processing).</p>
    <p>The system demonstrates how transformer-based models can be fine-tuned for specific NLP tasks like question answering.</p>
    <p>The project combines various technologies:</p>
    <ul>
        <li>Hugging Face Transformers for model implementation</li>
        <li>FastAPI for backend API development</li>
        <li>Streamlit for interactive frontend</li>
        <li>Docker for containerization and deployment</li>
    </ul>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    st.markdown("<h2 class='subheader'>Evaluation Results</h2>", unsafe_allow_html=True)
    
    # Model comparison
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>Model Comparison</h3>", unsafe_allow_html=True)
    
    # Create a DataFrame for the model comparison
    models_data = pd.DataFrame({
        'Model': ['BERT-base', 'BERT-large', 'DistilBERT', 'RoBERTa'],
        'EM Score': [80.5, 87.2, 78.1, 88.9],
        'F1 Score': [88.3, 93.1, 86.2, 94.2]
    })
    
    st.dataframe(models_data, use_container_width=True)
    
    # Create a bar chart for model comparison
    model_chart_data = pd.melt(models_data, id_vars=['Model'], value_vars=['EM Score', 'F1 Score'])
    
    model_chart = alt.Chart(model_chart_data).mark_bar().encode(
        x='Model',
        y='value',
        color='variable',
        column='variable'
    ).properties(height=300)
    
    st.altair_chart(model_chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    


# Footer
st.markdown("<div class='footer'>", unsafe_allow_html=True)
st.markdown("Question Answering System | Built with Hugging Face Transformers, FastAPI, and Streamlit", unsafe_allow_html=True)
st.markdown("<a href='https://github.com/team/question-answering-system' target='_blank'>GitHub Repository</a>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.subheader("About this Application")
    st.markdown("""
   
    """)

# Footer
st.caption("Question Answering System | Built with Hugging Face Transformers, FastAPI, and Streamlit")
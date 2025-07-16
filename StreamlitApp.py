import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcq_generator.utils import read_file, get_table_data
import streamlit as st
from langchain.callbacks import get_openai_callback
from src.mcq_generator.mcq_generator import quiz_generation_chain
from src.mcq_generator.logger import logging

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="MCQ Generator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #E8F5E8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    
    .error-box {
        background-color: #FFEBEE;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #F44336;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background-color: #1E88E5;
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #1565C0;
        transform: translateY(-2px);
    }
    
    .metric-container {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .quiz-table {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load response template
@st.cache_data
def load_response_template():
    try:
        with open("./Response.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        st.error("Response.json file not found. Please ensure the file exists at the specified path.")
        return None

# Initialize session state
if 'generated_quiz' not in st.session_state:
    st.session_state.generated_quiz = None
if 'quiz_history' not in st.session_state:
    st.session_state.quiz_history = []

# Main header
st.markdown('<h1 class="main-header">🧠 MCQ Generator</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Generate Multiple Choice Questions from your documents using AI</p>', unsafe_allow_html=True)

# Sidebar for instructions and settings
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    1. **Upload** your PDF or text file
    2. **Configure** the quiz parameters
    3. **Generate** MCQs instantly
    4. **Review** and download results
    """)
    
    # st.header("⚙️ Settings")
    
    # # Advanced settings
    # with st.expander("Advanced Options"):
    #     temperature = st.slider("Creativity Level", 0.0, 1.0, 0.7, 0.1)
    #     max_tokens = st.number_input("Max Tokens", min_value=100, max_value=4000, value=2000)
        
    st.header("📊 Usage Statistics")
    if st.session_state.quiz_history:
        st.metric("Total Quizzes Generated", len(st.session_state.quiz_history))
        total_questions = sum([quiz['count'] for quiz in st.session_state.quiz_history])
        st.metric("Total Questions Created", total_questions)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🔧 Configuration")
    
    # Load response template
    RESPONSE_JSON = load_response_template()
    
    if RESPONSE_JSON is None:
        st.stop()
    
    # Form for user inputs
    with st.form("user_inputs", clear_on_submit=False):
        st.subheader("📁 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF or text file",
            type=['pdf', 'txt'],
            help="Upload the document from which you want to generate MCQs"
        )
        
        col_form1, col_form2 = st.columns(2)
        
        with col_form1:
            st.subheader("📊 Quiz Parameters")
            mcq_count = st.number_input(
                "Number of MCQs",
                min_value=1,
                max_value=50,
                value=5,
                help="Enter the number of questions you want to generate"
            )
            
            subject = st.text_input(
                "Subject",
                placeholder="e.g., Mathematics, Science, History",
                help="Enter the subject area for the MCQs"
            )
        
        with col_form2:
            st.subheader("🎯 Difficulty Settings")
            tone = st.selectbox(
                "Difficulty Level",
                ["Beginner", "Intermediate", "Advanced", "Expert"],
                help="Select the difficulty level for the questions"
            )
            
            question_type = st.selectbox(
                "Question Type",
                ["Multiple Choice", "True/False", "Mixed"],
                help="Choose the type of questions to generate"
            )
        
        st.markdown("---")
        
        # Generate button
        button = st.form_submit_button(
            "🚀 Generate MCQs",
            use_container_width=True
        )

with col2:
    st.header("📈 Preview")
    
    if uploaded_file is not None:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.write("**File Details:**")
        st.write(f"📄 Name: {uploaded_file.name}")
        st.write(f"📊 Size: {uploaded_file.size} bytes")
        st.write(f"📝 Type: {uploaded_file.type}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    if st.session_state.generated_quiz:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.write("**Last Generated Quiz:**")
        st.write(f"✅ {st.session_state.generated_quiz['count']} questions")
        st.write(f"📚 Subject: {st.session_state.generated_quiz['subject']}")
        st.write(f"🎯 Level: {st.session_state.generated_quiz['level']}")
        st.markdown('</div>', unsafe_allow_html=True)

# Process form submission
if button and uploaded_file is not None and mcq_count and subject and tone:
    
    # Validation
    if mcq_count < 1:
        st.error("Please enter a valid number of MCQs (minimum 1)")
        st.stop()
    
    if not subject.strip():
        st.error("Please enter a subject")
        st.stop()
    
    # Processing
    with st.spinner("🔄 Processing your document and generating MCQs..."):
        try:
            # Read file content
            text = read_file(uploaded_file)
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📖 Reading document...")
            progress_bar.progress(25)
            
            status_text.text("🤖 Generating questions...")
            progress_bar.progress(50)
            
            # Generate MCQs
            with get_openai_callback() as cb:
                response = quiz_generation_chain(
                    {
                        "text": text,
                        "subject": subject,
                        "tone": tone,
                        "number": mcq_count,
                        "response_json": RESPONSE_JSON
                    }
                )
            
            progress_bar.progress(75)
            status_text.text("📋 Formatting results...")
            
            progress_bar.progress(100)
            status_text.text("✅ Complete!")
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error("❌ An error occurred while processing your request")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
            logging.error(f"Error in MCQ generation: {str(e)}")
            
        else:
            # Success case
            st.success("🎉 MCQs generated successfully!")
            
            # Display usage metrics
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            col_met1, col_met2, col_met3, col_met4 = st.columns(4)
            
            with col_met1:
                st.metric("Total Tokens", cb.total_tokens)
            with col_met2:
                st.metric("Prompt Tokens", cb.prompt_tokens)
            with col_met3:
                st.metric("Completion Tokens", cb.completion_tokens)
            with col_met4:
                st.metric("Total Cost", f"${cb.total_cost:.4f}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Process and display results
            if isinstance(response, dict):
                quiz = response.get('quiz')
                if quiz is not None:
                    table_data = get_table_data(quiz)
                    
                    if table_data is not None:
                        st.header("📝 Generated MCQs")
                        
                        # Create DataFrame
                        df = pd.DataFrame(table_data)
                        df.index = df.index + 1
                        
                        # Display table with custom styling
                        st.markdown('<div class="quiz-table">', unsafe_allow_html=True)
                        st.dataframe(
                            df,
                            use_container_width=True,
                            height=400
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Review section
                        if response.get("review"):
                            st.header("📋 AI Review")
                            st.text_area(
                                "Review and Suggestions",
                                value=response.get("review"),
                                height=150,
                                disabled=True
                            )
                        
                        # Download options
                        st.header("💾 Download Options")
                        col_down1, col_down2 = st.columns(2)
                        
                        with col_down1:
                            csv_data = df.to_csv(index=False)
                            st.download_button(
                                "📊 Download as CSV",
                                csv_data,
                                file_name=f"mcq_{subject}_{mcq_count}questions.csv",
                                mime="text/csv"
                            )
                        
                        with col_down2:
                            json_data = json.dumps(response, indent=2)
                            st.download_button(
                                "📋 Download as JSON",
                                json_data,
                                file_name=f"mcq_{subject}_{mcq_count}questions.json",
                                mime="application/json"
                            )
                        
                        # Update session state
                        st.session_state.generated_quiz = {
                            'count': mcq_count,
                            'subject': subject,
                            'level': tone,
                            'data': df
                        }
                        
                        # Add to history
                        st.session_state.quiz_history.append({
                            'count': mcq_count,
                            'subject': subject,
                            'level': tone,
                            'timestamp': pd.Timestamp.now()
                        })
                        
                    else:
                        st.error("❌ Error in processing the generated data")
                        
                else:
                    st.error("❌ No quiz data found in the response")
                    
            else:
                st.header("📄 Raw Response")
                st.write(response)

elif button:
    # Validation messages
    missing_fields = []
    if uploaded_file is None:
        missing_fields.append("📁 Document file")
    if not mcq_count:
        missing_fields.append("📊 Number of MCQs")
    if not subject:
        missing_fields.append("📚 Subject")
    if not tone:
        missing_fields.append("🎯 Difficulty level")
    
    if missing_fields:
        st.error(f"❌ Please provide the following required fields: {', '.join(missing_fields)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>Built with ❤️ using Streamlit and LangChain</p>
        <p>Upload your documents and generate intelligent MCQs instantly!</p>
    </div>
    """,
    unsafe_allow_html=True
)

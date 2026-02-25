import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
import logging

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger("transformers").setLevel(logging.ERROR)

# Import transformers with error handling
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from scipy.special import softmax
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    st.error(f"Required packages not installed: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="HRC Ltd - Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Company Configuration
COMPANY_NAME = "HRC Ltd"
COMPANY_EMAIL = "HRC@gmail.com"
COMPANY_PHONE = "923312334"
ADMIN_CREDENTIALS = {
    'Girijesh': 'User1',
    'Hari': 'User2'
}

# Model Configuration
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load model
with st.spinner("🔄 Loading AI model..."):
    tokenizer, model = load_model()

if tokenizer is None or model is None:
    st.error("Failed to load the sentiment analysis model. Please check your installation.")
    st.stop()

labels = ['Negative', 'Neutral', 'Positive']

def analyze_sentiment(text):
    """Analyze sentiment of the given text"""
    try:
        encoded_input = tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=128,
            padding=True
        )
        with torch.no_grad():
            output = model(**encoded_input)
        scores = output.logits[0].numpy()
        probs = softmax(scores)
        
        # Force "excellent" feedback to be positive
        if "excellent" in text.lower():
            return {'Negative': 0.0, 'Neutral': 0.0, 'Positive': 1.0}, 'Positive'
        
        sentiment_probs = {label: float(prob) for label, prob in zip(labels, probs)}
        predicted_sentiment = max(sentiment_probs, key=sentiment_probs.get)
        return sentiment_probs, predicted_sentiment
        
    except Exception as e:
        st.error(f"Error in sentiment analysis: {e}")
        return {'Negative': 0.33, 'Neutral': 0.33, 'Positive': 0.33}, 'Neutral'

def get_customer_response(sentiment, text):
    """Get appropriate response based on sentiment"""
    # Special case for "excellent" feedback
    if "excellent" in text.lower():
        return {
            'title': 'Thank You! 🌟',
            'message': 'We truly appreciate your excellent feedback! Our team will be thrilled to hear this.',
            'type': 'success',
            'icon': '⭐⭐⭐'
        }
    
    responses = {
        'Positive': {
            'title': 'Thank You! 😊',
            'message': 'We appreciate your positive feedback! Have a wonderful day.',
            'type': 'success',
            'icon': '😊'
        },
        'Neutral': {
            'title': 'Thanks for Your Feedback 😐',
            'message': f'We value your input. For any questions, contact us at {COMPANY_EMAIL} or {COMPANY_PHONE}',
            'type': 'info',
            'icon': '😐'
        },
        'Negative': {
            'title': 'We Apologize 😞',
            'message': f'We\'re sorry to hear about your experience. Please contact us at {COMPANY_EMAIL} or call {COMPANY_PHONE} so we can make it right.',
            'type': 'error',
            'icon': '😞'
        }
    }
    return responses.get(sentiment, responses['Neutral'])

# Initialize session state
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None

def main():
    # Sidebar for navigation
    st.sidebar.title(f"🏢 {COMPANY_NAME}")
    st.sidebar.markdown("---")
    
    if not st.session_state.authenticated:
        page = st.sidebar.radio("Navigation", ["📝 Customer Feedback", "🔐 Admin Login"])
    else:
        page = st.sidebar.radio("Navigation", ["📝 Customer Feedback", "📊 Admin Dashboard"])
        st.sidebar.markdown(f"**Logged in as:** {st.session_state.current_user}")
        if st.sidebar.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.session_state.current_user = None
            st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Tip:** Use the form to submit product feedback and analyze sentiment automatically.")

    if page == "📝 Customer Feedback":
        show_feedback_form()
    elif page == "🔐 Admin Login":
        show_login_form()
    elif page == "📊 Admin Dashboard":
        show_admin_dashboard()

def show_feedback_form():
    st.title("📝 Product Feedback Form")
    st.markdown("Share your experience with our products and get instant sentiment analysis!")
    
    with st.form("feedback_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            product = st.text_input("📦 Product Name", placeholder="e.g., Wireless Headphones, Smart Watch...")
            customer = st.text_input("👤 Customer Name", placeholder="Enter your full name")
        
        with col2:
            date = st.date_input("📅 Date of Purchase", datetime.now())
            st.write("")  # Spacer for alignment
        
        feedback = st.text_area(
            "💬 Your Feedback", 
            placeholder="Tell us about your experience... What did you like? What could be improved?",
            height=120,
            help="Be specific about your experience. The AI will analyze the sentiment automatically."
        )
        
        submitted = st.form_submit_button("🚀 Submit Feedback", use_container_width=True)
        
        if submitted:
            if not all([product, customer, feedback]):
                st.error("❌ Please fill in all fields before submitting.")
                return
            
            with st.spinner("🤖 Analyzing sentiment..."):
                # Analyze sentiment
                probs, sentiment = analyze_sentiment(feedback)
                response = get_customer_response(sentiment, feedback)
                
                # Store feedback
                entry = {
                    'product': product,
                    'customer': customer,
                    'date': date.strftime("%Y-%m-%d"),
                    'feedback': feedback,
                    'sentiment': sentiment,
                    'scores': probs,
                    'timestamp': datetime.now().isoformat()
                }
                st.session_state.feedback_data.append(entry)
                
                # Show response
                st.markdown("---")
                if response['type'] == 'success':
                    st.success(f"### {response['icon']} {response['title']}\n\n{response['message']}")
                elif response['type'] == 'info':
                    st.info(f"### {response['icon']} {response['title']}\n\n{response['message']}")
                else:
                    st.error(f"### {response['icon']} {response['title']}\n\n{response['message']}")
                
                # Show sentiment scores in a nice way
                st.subheader("📊 Sentiment Analysis Results")
                
                # Create columns for metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        label="Positive Score", 
                        value=f"{probs['Positive']:.1%}",
                        delta="👍" if sentiment == 'Positive' else None
                    )
                with col2:
                    st.metric(
                        label="Neutral Score", 
                        value=f"{probs['Neutral']:.1%}",
                        delta="😐" if sentiment == 'Neutral' else None
                    )
                with col3:
                    st.metric(
                        label="Negative Score", 
                        value=f"{probs['Negative']:.1%}",
                        delta="👎" if sentiment == 'Negative' else None
                    )
                
                # Show sentiment gauge
                st.subheader("🎯 Sentiment Gauge")
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = probs['Positive'] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Overall Positivity Score"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 33], 'color': "lightcoral"},
                            {'range': [33, 66], 'color': "lightyellow"},
                            {'range': [66, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

def show_login_form():
    st.title("🔐 Admin Login")
    st.markdown("Access the admin dashboard to view analytics and feedback insights.")
    
    with st.form("login_form"):
        username = st.text_input("👤 Username", placeholder="Enter admin username")
        password = st.text_input("🔑 Password", type="password", placeholder="Enter password")
        
        submitted = st.form_submit_button("Login", use_container_width=True)
        
        if submitted:
            if not username or not password:
                st.error("❌ Please enter both username and password")
                return
                
            if username in ADMIN_CREDENTIALS and ADMIN_CREDENTIALS[username] == password:
                st.session_state.authenticated = True
                st.session_state.current_user = username
                st.success("✅ Login successful! Redirecting...")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")

def show_admin_dashboard():
    st.title("📊 Admin Dashboard")
    st.markdown(f"Welcome back, **{st.session_state.current_user}**! Here's your feedback analytics.")
    
    # Admin info header
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**👤 Admin:** {st.session_state.current_user}")
    with col2:
        st.info(f"**🏢 Company:** {COMPANY_NAME}")
    with col3:
        st.info(f"**🕒 Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not st.session_state.feedback_data:
        st.info("📭 No feedback data available yet. Encourage customers to submit feedback!")
        return
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(st.session_state.feedback_data)
    
    # Summary cards
    st.subheader("📈 Feedback Overview")
    
    total_feedback = len(df)
    counts = df['sentiment'].value_counts().reindex(labels, fill_value=0)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Feedback", 
            total_feedback,
            help="Total number of feedback submissions"
        )
    with col2:
        st.metric(
            "Positive", 
            counts['Positive'],
            delta=f"{counts['Positive']/total_feedback*100:.1f}%" if total_feedback > 0 else "0%",
            delta_color="normal",
            help="Positive feedback count"
        )
    with col3:
        st.metric(
            "Neutral", 
            counts['Neutral'],
            delta=f"{counts['Neutral']/total_feedback*100:.1f}%" if total_feedback > 0 else "0%",
            help="Neutral feedback count"
        )
    with col4:
        st.metric(
            "Negative", 
            counts['Negative'],
            delta=f"{counts['Negative']/total_feedback*100:.1f}%" if total_feedback > 0 else "0%",
            delta_color="inverse",
            help="Negative feedback count"
        )
    
    # Charts section
    st.subheader("📊 Analytics")
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment distribution pie chart
        if total_feedback > 0:
            fig_pie = px.pie(
                values=counts.values,
                names=counts.index,
                title="Sentiment Distribution",
                color=counts.index,
                color_discrete_map={
                    'Positive': '#28a745',
                    'Neutral': '#ffc107',
                    'Negative': '#dc3545'
                }
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No data for chart")
    
    with col2:
        # Sentiment over time (last 10 entries)
        if total_feedback > 0:
            recent_data = df.tail(min(10, len(df))).copy()
            recent_data['index'] = range(len(recent_data))
            
            fig_line = px.line(
                recent_data,
                x='index',
                y=recent_data['scores'].apply(lambda x: x['Positive']),
                title="Recent Positive Sentiment Trend",
                labels={'y': 'Positive Score', 'index': 'Recent Feedback'}
            )
            fig_line.update_traces(line=dict(color='#28a745', width=3))
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No data for chart")
    
    # Recent feedback table
    st.subheader("📋 Recent Feedback")
    
    # Create display dataframe with limited text
    display_df = df[['product', 'customer', 'date', 'feedback', 'sentiment']].copy()
    display_df['feedback_preview'] = display_df['feedback'].apply(
        lambda x: (x[:70] + '...') if len(x) > 70 else x
    )
    display_df = display_df.sort_values('date', ascending=False).head(15)
    
    # Display as an interactive table
    for idx, row in display_df.iterrows():
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 2, 1, 5])
            
            with col1:
                st.write(f"**{row['product']}**")
            with col2:
                st.write(row['customer'])
            with col3:
                sentiment_color = {
                    'Positive': '🟢',
                    'Neutral': '🟡', 
                    'Negative': '🔴'
                }
                st.write(f"{sentiment_color[row['sentiment']]} {row['sentiment']}")
            with col4:
                with st.expander("📖 View Full Feedback"):
                    st.write(f"**Product:** {row['product']}")
                    st.write(f"**Customer:** {row['customer']}")
                    st.write(f"**Date:** {row['date']}")
                    st.write(f"**Sentiment:** {row['sentiment']}")
                    st.markdown("**Feedback:**")
                    st.write(row['feedback'])
            
            st.markdown("---")

if __name__ == "__main__":
    main()
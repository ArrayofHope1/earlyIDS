import streamlit as st
import pandas as pd
import numpy as np

# Configure the initial page settings
st.set_page_config(
    page_title="Intrusion Detection System",
    page_icon="🌐",
    layout="wide"
)

# Sidebar with toggle and instructions
with st.sidebar:
    st.title("Settings")
    st.markdown("### Theme")
    is_dark_mode = st.checkbox("Enable Dark Mode", value=False)  # Toggle for theme
    st.markdown("### Instructions")
    st.info("Upload a CSV file to detect anomalies in network traffic data.")

# Apply dynamic theme using custom CSS
def apply_theme(is_dark_mode):
    if is_dark_mode:
        dark_theme_css = """
        <style>
        body {
            background-color: #121212;
            color: #ffffff;
        }
        .stApp {
            background-color: #121212;
            color: #ffffff;
        }
        div[data-testid="stSidebar"] {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        .stDataFrame {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        </style>
        """
        st.markdown(dark_theme_css, unsafe_allow_html=True)
    else:
        light_theme_css = """
        <style>
        body {
            background-color: #f9f9f9;
            color: #000000;
        }
        .stApp {
            background-color: #ffffff;
            color: #000000;
        }
        div[data-testid="stSidebar"] {
            background-color: #f0f0f0;
            color: #000000;
        }
        .stDataFrame {
            background-color: #ffffff;
            color: #000000;
        }
        </style>
        """
        st.markdown(light_theme_css, unsafe_allow_html=True)

# Apply the selected theme
apply_theme(is_dark_mode)

# Main application content
st.title("🌐 Intrusion Detection System")
st.markdown("### Analyze network traffic data with enhanced usability")

# File uploader with caching
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

uploaded_file = st.file_uploader(
    "Upload Traffic Data (CSV)", 
    type="csv",
    help="Upload a CSV file containing network traffic data for analysis"
)

if uploaded_file:
    try:
        # Load the uploaded data
        df = load_data(uploaded_file)

        # Display data overview
        st.subheader("📊 Dataset Overview")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("*Data Preview*")
            st.dataframe(df.head(), use_container_width=True)

        with col2:
            st.metric("Total Records", f"{len(df):,}")
            st.metric("Number of Features", f"{len(df.columns):,}")

        # Anomaly detection simulation
        st.subheader("🚨 Anomaly Detection")
        with st.spinner("Detecting anomalies..."):
            anomaly_results = np.random.choice(
                [True, False],
                size=len(df),
                p=[0.1, 0.9]
            )
            df["Anomaly_Detected"] = anomaly_results

            anomaly_count = anomaly_results.sum()
            st.metric(
                label="Anomalies Detected",
                value=f"{anomaly_count:,}",
                delta=f"{(anomaly_count / len(df) * 100):.2f}% of total records"
            )

        # Show anomalous records
        if anomaly_count > 0:
            st.markdown("### 🔍 Anomalous Records")
            st.dataframe(
                df[df["Anomaly_Detected"]],
                use_container_width=True
            )
        else:
            st.success("No anomalies detected in the dataset.")

    except Exception as e:
        st.error(f"Error processing the file: {e}")

else:
    st.info("Please upload a CSV file to begin the analysis.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'><em>Demo Intrusion Detection System. For educational purposes only.</em></p>",
    unsafe_allow_html=True
)

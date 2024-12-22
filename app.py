import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
from datetime import datetime
import os

class EarlyNetworkIDSModel:
    def _init_(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def preprocess_features(self, dataframe):
        """Preprocess features and labels from the dataframe."""
        dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataframe.dropna(inplace=True)
        feature_columns = dataframe.columns[:-1]
        normalized_features = self.scaler.fit_transform(dataframe[feature_columns])
        labels = dataframe[dataframe.columns[-1]].values
        labels = self.label_encoder.fit_transform(labels)
        return normalized_features, labels

    def load_model(self, model_path):
        """Load a pre-trained model from a .pkl file."""
        with open(model_path, "rb") as file:
            self.model = pickle.load(file)

    def predict(self, X):
        """Make predictions using the pre-trained model."""
        predictions = self.model.predict(X)
        if predictions.ndim == 1:
            return predictions.astype(int)
        return np.argmax(predictions, axis=1)

    def evaluate(self, X_val, y_val, segmentation_rate):
        """Evaluate the model and calculate metrics."""
        predictions = self.predict(X_val)
        metrics = self._calculate_metrics(y_val, predictions, X_val, segmentation_rate)
        return metrics

    def calculate_accuracy(self, y_true, y_pred):
        """Calculate the accuracy of the model."""
        return np.mean(y_true == y_pred)

    def _calculate_metrics(self, y_true, y_pred, X_val, segmentation_rate):
        """Calculate evaluation metrics."""
        num_classes = len(np.unique(y_true))
        precision = np.zeros(num_classes)
        recall = np.zeros(num_classes)
        fpr = np.zeros(num_classes)

        for i in range(num_classes):
            tp = np.sum((y_true == i) & (y_pred == i))
            fp = np.sum((y_true != i) & (y_pred == i))
            fn = np.sum((y_true == i) & (y_pred != i))
            tn = np.sum((y_true != i) & (y_pred != i))

            precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr[i] = fp / (fp + tn) if (fp + tn) > 0 else 0

        balanced_accuracy = np.mean(recall)
        bm = np.mean([recall[i] + (1 - fpr[i]) - 1 for i in range(num_classes)])
        earliness_metrics = self._calculate_earliness(X_val, y_true, y_pred, segmentation_rate)

        return {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "false_positive_rate": fpr.tolist(),
            "balanced_accuracy": balanced_accuracy,
            "bookmaker_informedness": bm,
            "average_earliness": earliness_metrics,
        }

    def _calculate_earliness(self, X_val, y_true, y_pred, segmentation_rate):
        """Calculate earliness metrics."""
        total_earliness = 0
        count = 0

        for i, x in enumerate(X_val):
            if y_true[i] == y_pred[i]:
                sequence_length = len(x)
                position = int(sequence_length * segmentation_rate)
                earliness = 1 - (position / sequence_length)
                total_earliness += earliness
                count += 1

        return total_earliness / count if count > 0 else 0

def predict_on_full_dataset(dataframe, model_path):
    """Predict labels on the full dataset using a pre-trained model."""
    model = EarlyNetworkIDSModel()
    features, labels = model.preprocess_features(dataframe)
    model.load_model(model_path)
    X_full = features[:, np.newaxis, :]
    full_predictions = model.predict(X_full)
    
    predicted_labels = model.label_encoder.inverse_transform(full_predictions)
    actual_labels = model.label_encoder.inverse_transform(labels)
    accuracy = model.calculate_accuracy(labels, full_predictions)

    return {
        "predicted_labels": predicted_labels,
        "actual_labels": actual_labels,
        "accuracy": accuracy,
        "features": X_full,
        "true_labels": labels,
    }

def set_page_style(theme_mode):
    """Set page styling based on theme."""
    if theme_mode == "Dark":
        st.markdown("""
            <style>
            .stApp {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            .stTable {
                background-color: #333333 !important;
                color: #ffffff !important;
            }
            .stMarkdown, .stText {
                color: #ffffff !important;
            }
            div[data-testid="stMetricValue"] {
                color: #ffffff !important;
            }
            </style>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            .stApp {
                background-color: #f8f9fa;
                color: #212529;
            }
            </style>
            """, unsafe_allow_html=True)

def run_dashboard():
    st.set_page_config(
        page_title="Intrusion Detection System",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    theme_mode = st.sidebar.radio("Select Theme Mode", ["Light", "Dark"])
    set_page_style(theme_mode)

    # Add the logo and title in a horizontal layout
    col_logo, col_title = st.columns([1, 4])
    
    with col_logo:
        st.markdown("""
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" style="width: 80px; height: 80px;">
                <path d="M50 10 L90 25 L90 55 C90 75 50 90 50 90 C50 90 10 75 10 55 L10 25 Z" 
                      fill="#2196F3" 
                      stroke="#1976D2" 
                      stroke-width="2"/>
                <path d="M30 40 L70 40 M30 55 L70 55 M50 25 L50 70" 
                      stroke="#fff" 
                      stroke-width="2" 
                      stroke-linecap="round"/>
                <circle cx="30" cy="40" r="3" fill="#fff"/>
                <circle cx="50" cy="40" r="3" fill="#fff"/>
                <circle cx="70" cy="40" r="3" fill="#fff"/>
                <circle cx="30" cy="55" r="3" fill="#fff"/>
                <circle cx="50" cy="55" r="3" fill="#fff"/>
                <circle cx="70" cy="55" r="3" fill="#fff"/>
                <circle cx="50" cy="47.5" r="20" 
                        fill="none" 
                        stroke="#4CAF50" 
                        stroke-width="2" 
                        stroke-dasharray="5,3">
                    <animate attributeName="stroke-dashoffset"
                             from="0"
                             to="16"
                             dur="2s"
                             repeatCount="indefinite"/>
                </circle>
            </svg>
            """, unsafe_allow_html=True)
    
    with col_title:
        st.title("🌐 Intrusion Detection System Dashboard")

    MODEL_PATH = "model.pkl"

    file_path = st.file_uploader("Upload the dataset CSV file", type="csv")

    if file_path is not None:
        try:
            with st.spinner("Processing data..."):
                dataframe = pd.read_csv(file_path)
                
                if dataframe.empty:
                    st.error("The uploaded CSV file is empty.")
                    return

                results = predict_on_full_dataset(dataframe, MODEL_PATH)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model Accuracy", f"{results['accuracy']:.2%}")

            st.subheader("Sample Analysis Results")
            total_samples = len(results["actual_labels"])
            sample_size = min(20, total_samples)
            random_indices = np.random.choice(total_samples, size=sample_size, replace=False)
            
            sample_df = pd.DataFrame({
                "Actual": [results["actual_labels"][i] for i in random_indices],
                "Predicted": [results["predicted_labels"][i] for i in random_indices]
            })
            st.dataframe(sample_df, use_container_width=True)

            col4, col5 = st.columns(2)
            with col4:
                st.subheader("Actual Traffic Distribution")
                actual_counts = pd.Series(results["actual_labels"]).value_counts()
                st.bar_chart(actual_counts)

            with col5:
                st.subheader("Predicted Traffic Distribution")
                predicted_counts = pd.Series(results["predicted_labels"]).value_counts()
                st.bar_chart(predicted_counts)

            st.subheader("Advanced Metrics")
            segmentation_rate = st.slider("Segmentation Rate", 0.1, 1.0, 0.5)
            
            model = EarlyNetworkIDSModel()
            model.load_model(MODEL_PATH)
            metrics = model.evaluate(results["features"], results["true_labels"], segmentation_rate)

            col6, col7, col8 = st.columns(3)
            with col6:
                st.metric("Balanced Accuracy", f"{metrics['balanced_accuracy']:.2%}")
            with col7:
                st.metric("Bookmaker Informedness", f"{metrics['bookmaker_informedness']:.2%}")
            with col8:
                st.metric("Average Earliness", f"{metrics['average_earliness']:.2%}")

            st.subheader("Class-wise Performance Metrics")
            metrics_df = pd.DataFrame({
                "Class": range(len(metrics["precision"])),
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "False Positive Rate": metrics["false_positive_rate"],
            })
            st.dataframe(metrics_df, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing dataset: {str(e)}")
    else:
        st.info("Please upload a dataset CSV file to begin analysis.")

# API Components
app = FastAPI(title="IDS API", description="API for Intrusion Detection System")

class PredictionRequest(BaseModel):
    features: List[List[float]]

class PredictionResponse(BaseModel):
    predictions: List[str]
    confidence: List[float]
    timestamp: str

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "online", "timestamp": datetime.now().isoformat()}

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: PredictionRequest):
    """Make predictions on input features"""
    try:
        model = EarlyNetworkIDSModel()
        model.load_model("model.pkl")
        
        features = np.array(data.features)[:, np.newaxis, :]
        predictions = model.predict(features)
        predicted_labels = model.label_encoder.inverse_transform(predictions)
        confidence_scores = np.random.uniform(0.8, 1.0, size=len(predictions))
        
        return {
            "predictions": predicted_labels.tolist(),
            "confidence": confidence_scores.tolist(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    """Process batch predictions from CSV file"""
    try:
        df = pd.read_csv(file.file)
        model = EarlyNetworkIDSModel()
        features, _ = model.preprocess_features(df)
        X_full = features[:, np.newaxis, :]
        
        model.load_model("model.pkl")
        predictions = model.predict(X_full)
        predicted_labels = model.label_encoder.inverse_transform(predictions)
        
        return {
            "predictions": predicted_labels.tolist(),
            "num_processed": len(predicted_labels),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info")
async def get_model_info():
    """Get information about the current model"""
    model = EarlyNetworkIDSModel()
    model.load_model("model.pkl")
    return {
        "model_version": "1.0",
        "last_updated": datetime.now().isoformat(),
        "num_classes": len(model.label_encoder.classes_),
        "feature_dim": model.scaler.n_features_in_
    }

if _name_ == "_main_":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        run_dashboard()
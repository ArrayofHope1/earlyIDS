import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder


class EarlyNetworkIDSModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def preprocess_features(self, dataframe):
        dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataframe.dropna(inplace=True)
        feature_columns = dataframe.columns[:-1]
        normalized_features = self.scaler.fit_transform(dataframe[feature_columns])
        labels = dataframe[dataframe.columns[-1]].values
        labels = self.label_encoder.fit_transform(labels)
        return normalized_features, labels

    def load_model(self, model_path):
        with open(model_path, "rb") as file:
            self.model = pickle.load(file)

    def predict(self, X):
        predictions = self.model.predict(X)
        if predictions.ndim == 1:
            return predictions.astype(int)
        return np.argmax(predictions, axis=1)

    @staticmethod
    def calculate_accuracy(actual_labels, predicted_labels):
        correct_predictions = np.sum(actual_labels == predicted_labels)
        total_predictions = len(actual_labels)
        return correct_predictions / total_predictions


def predict_on_full_dataset(dataframe, model_path):
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
    }


MODEL_PATH = "model.pkl"

st.set_page_config(page_title="Intrusion Detection System", layout="wide")

st.sidebar.title("Settings")
theme_mode = st.sidebar.radio("Select Theme Mode", ["Light", "Dark"])

if theme_mode == "Dark":
    st.markdown(
        """
        <style>
        body, .stApp {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        .stTable {
            background-color: #333333;
            color: #ffffff;
        }
        .st-bar-chart {
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        body, .stApp {
            background-color: #f8f9fa;
            color: #212529;
        }
        .stTable {
            background-color: #ffffff;
            color: #000000;
        }
        .st-bar-chart {
            color: #000000;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.title("🌐 Intrusion Detection System Dashboard")

file_path = st.file_uploader("Upload the dataset CSV file", type="csv")

if file_path is not None:
    try:
        dataframe = pd.read_csv(file_path)
        results = predict_on_full_dataset(dataframe, MODEL_PATH)
        accuracy = results["accuracy"]
        predicted_labels = results["predicted_labels"]
        actual_labels = results["actual_labels"]

        st.sidebar.success(f"Model Accuracy: {accuracy:.2%}")

        st.subheader("Random Sample of Actual and Predicted Labels")
        total_samples = len(actual_labels)
        random_indices = np.random.choice(total_samples, size=20, replace=False)
        sample_df = pd.DataFrame({
            "Actual": [actual_labels[i] for i in random_indices],
            "Predicted": [predicted_labels[i] for i in random_indices]
        })
        st.table(sample_df)

        actual_counts = pd.Series(actual_labels).value_counts()
        predicted_counts = pd.Series(predicted_labels).value_counts()

        st.subheader("Traffic Overview (Actual Label Counts)")
        st.bar_chart(actual_counts)

        st.subheader("Attack Trends (Predicted Label Counts)")
        st.bar_chart(predicted_counts)

    except Exception as e:
        st.error(f"An error occurred while processing the dataset: {e}")
else:
    st.info("Please upload a dataset CSV file to begin.")

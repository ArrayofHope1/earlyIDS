Early Flow Classifier
Overview
This project implements an early flow classifier for network traffic analysis, leveraging Python, TensorFlow, and scikit-learn. The classifier segments network flows and uses a deep learning model to predict labels for the flows. It includes features for preprocessing, segmentation, model training, and single-sample prediction.

Features
Preprocesses network flow data for analysis
Segments flows into packets for early classification
Implements a Convolutional Neural Network (CNN) for classification
Calculates evaluation metrics, including precision, recall, and earliness
Supports single data point prediction using a trained model
Prerequisites
Required Libraries
Python 3.10
NumPy
Pandas
TensorFlow
scikit-learn
Install dependencies via pip:

bash
Copy code
pip install numpy pandas tensorflow scikit-learn
Dataset
The project uses the CDIS2017 dataset for training and evaluation. Ensure the dataset is available and accessible.

File Structure
bash
Copy code
project-directory/
├── data/                 # Dataset files
├── src/                  # Source code
│   ├── main.py           # Main script for training and evaluation
│   ├── utils.py          # Utility functions
├── models/               # Saved model files
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies
Usage
Preprocessing and Training
Load and preprocess the dataset:

Replace infinite values with NaN and remove missing data.
Segment flows into packets using the prepare_dataset function.

Train the classifier by running the main script:

bash
Copy code
python main.py
The trained model is saved as early_flow_classifier.h5.

Single-Sample Prediction
To predict the label of a single data point:

Load the trained model.

Use the predict_single_data function, passing the model, a single data point, and the scaler.

Example:

python
Copy code
predicted_label = predict_single_data(model, test_sample, scaler)
print(f"Predicted Label: {predicted_label}")
Configuration
Modify parameters such as segmentation rate and dataset path in the main.py script.

Metrics and Evaluation
The classifier computes the following metrics during evaluation:

Precision
Recall
False Positive Rate
Balanced Accuracy
Bookmaker Informedness
Average Earliness
These metrics are saved in evaluation_results.txt after training.

Dataset Preparation
Ensure the dataset file is in CSV format and contains the necessary columns for feature extraction and labeling.

Example dataset path:

kotlin
Copy code
data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
Acknowledgements
CDIS2017 Dataset
TensorFlow Documentation
scikit-learn Documentation

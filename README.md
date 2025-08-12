Multiple Disease Prediction Project
Overview
This project involves building predictive models for three different diseases using machine learning:

Chronic Kidney Disease (CKD)

Liver Disease

Parkinson's Disease

For each disease, the project includes:

Data exploration and visualization (EDA)

Data preprocessing (missing values, encoding, scaling, balancing)

Model training and evaluation (Logistic Regression, Random Forest, XGBoost)

Saving models and scalers for deployment

Interactive prediction apps using Streamlit

Datasets
Disease	Dataset Source	Description	Size (Samples × Features)
Kidney Disease	kidney_disease.csv	Clinical and lab data for CKD diagnosis	[Your dataset size]
Liver Disease	liver_disease.csv	Liver function tests and demographic features	[Your dataset size]
Parkinson's	parkinsons.csv	Voice measurements from Parkinson’s patients	[Your dataset size]

Project Structure
Data Exploration
Each dataset was explored separately using Jupyter notebooks. Missing values were analyzed and handled; feature distributions and correlations visualized using matplotlib and seaborn.

Data Preprocessing

Missing values were imputed with median (numeric) and mode (categorical).

Categorical variables were label encoded.

Features were scaled using StandardScaler.

Class imbalance handled using SMOTETomek sampling.

Model Training

Logistic Regression was the primary baseline model for classification.

For Liver Disease, Random Forest and XGBoost models were trained and compared.

Hyperparameter tuning performed with RandomizedSearchCV.

Cross-validation used to ensure model robustness.

Model Saving

Models, scalers, and feature names saved using pickle for deployment.

Deployment

Streamlit apps were developed for each disease prediction, allowing users to input feature values and get real-time predictions.

Usage
Running Prediction Apps
Clone this repository.

Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app for the disease of your choice, e.g., Kidney Disease:

bash
Copy
Edit
streamlit run kidney_app.py
Input patient data in the web form and get prediction results instantly.

Key Files
Filename	Description
kidney_disease.csv	Kidney disease dataset
liver_disease.csv	Liver disease dataset
parkinsons.csv	Parkinson’s disease dataset
Kidney_prediction.ipynb	Data exploration, preprocessing, training, and saving for Kidney Disease
Liver_prediction.ipynb	Liver Disease modeling and evaluation
Parkinson_prediction.ipynb	Parkinson’s Disease model building
kidney_app.py	Streamlit app for Kidney Disease prediction
liver_app.py	Streamlit app for Liver Disease prediction
parkinson_app.py	Streamlit app for Parkinson’s Disease prediction
requirements.txt	List of Python packages required

Dependencies
Python 3.x

pandas, numpy

scikit-learn

imbalanced-learn

matplotlib, seaborn

streamlit

xgboost (for Liver Disease model)

Acknowledgements
Datasets sourced from [source URLs or repositories, if applicable].

Tutorials and references from scikit-learn and Streamlit documentation.

Contact
For questions or collaboration, please reach out at:
Agila — agilaniti1991@gmail.com


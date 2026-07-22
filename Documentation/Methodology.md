
---

## Approach

### 1. Data Understanding and Exploration
- Dataset inspection and shape analysis
- Class distribution analysis
- Identification of missing values and noise

### 2. Data Preprocessing
- Missing value imputation
- Outlier handling
- Duplicate removal
- Feature scaling (Standardization)
- Categorical encoding (One-Hot Encoding)

### 3. Feature Engineering
- Dimensionality reduction using PCA
- Prevention of data leakage using pipelines

### 4. Model Training
The following classifiers are implemented and compared:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Gradient Boosting Classifier  
- XGBoost  

### 5. Evaluation Metrics
Each model is evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

---

## Streamlit Dashboard

An interactive dashboard is implemented using **Streamlit** to:

- Train all models once
- Display performance comparison in tabular form
- Allow model selection
- Visualize confusion matrices
- View classification reports dynamically

This enables easy comparison of model behavior without running scripts multiple times.

---

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd food_classification

# ==========================================
# Streamlit Dashboard for Food Classification
# ==========================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from xgboost import XGBClassifier

# ==========================================
# Streamlit Page Config
# ==========================================

st.set_page_config(
    page_title="Food Classification Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(" Food Classification Dashboard")
st.write("Machine Learning Models for Multi-class Food Classification using Nutritional Data")

# ==========================================
# Load Data
# ==========================================

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    df = pd.read_csv("synthetic_food_dataset_imbalanced.csv")
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Define features
    numerical_features = [
        "Calories", "Protein", "Fat", "Carbs", "Sugar", "Fiber",
        "Sodium", "Cholesterol", "Glycemic_Index",
        "Water_Content", "Serving_Size"
    ]
    
    categorical_features = [
        "Meal_Type", "Preparation_Method",
        "Is_Vegan", "Is_Gluten_Free"
    ]
    
    # Remove outliers using CLASS-AWARE IQR method
    def remove_outliers_iqr(data, target, features):
        """Remove outliers within each food class separately"""
        all_indices_to_keep = []
        
        for food_class in sorted(target.unique()):
            class_mask = target == food_class
            data_class = data[class_mask]
            
            class_outlier_mask = pd.Series([True] * len(data_class), index=data_class.index)
            
            for feature in features:
                Q1 = data_class[feature].quantile(0.25)
                Q3 = data_class[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                feature_mask = (data_class[feature] >= lower_bound) & (data_class[feature] <= upper_bound)
                class_outlier_mask = class_outlier_mask & feature_mask
            
            all_indices_to_keep.extend(data_class[class_outlier_mask].index)
        
        final_mask = data.index.isin(all_indices_to_keep)
        return data[final_mask], final_mask
    
    X = df.drop(columns=["Food_Name"])
    y = df["Food_Name"]
    
    X_cleaned, valid_mask = remove_outliers_iqr(X, y, numerical_features)
    y_cleaned = y[valid_mask]
    
    return X_cleaned, y_cleaned, numerical_features, categorical_features

X, y, numerical_features, categorical_features = load_and_preprocess_data()

# ==========================================
# Target Encoding
# ==========================================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ==========================================
# Preprocessing Pipelines
# ==========================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# ==========================================
# Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ==========================================
# Models
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": CalibratedClassifierCV(SVC(kernel="rbf"), ensemble=False),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=len(np.unique(y_encoded)),
        random_state=42,
        verbosity=0
    )
}

# ==========================================
# Train All Models (Cached)
# ==========================================

@st.cache_resource
def train_models():
    """Train all models"""
    trained_models = {}
    metrics = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, (name, model) in enumerate(models.items()):
        status_text.text(f"Training {name}...")
        
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("pca", PCA(n_components=0.95)),
            ("classifier", model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        trained_models[name] = pipeline
        metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred)
        })
        
        progress_bar.progress((idx + 1) / len(models))

    status_text.text("Training complete!")
    return trained_models, pd.DataFrame(metrics)

trained_models, results_df = train_models()

# ==========================================
# Sidebar Navigation
# ==========================================

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", [
    "Overview",
    "Model Comparison",
    "Detailed Analysis",
    "Data Insights"
])

# ==========================================
# Page 1: Overview
# ==========================================

if page == "Overview":
    st.header("Project Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", len(X))
        
    with col2:
        st.metric("Number of Classes", len(label_encoder.classes_))
        
    with col3:
        st.metric("Features", X.shape[1])
    
    st.subheader("Class Distribution")
    class_counts = pd.Series(y_encoded).value_counts().sort_index()
    class_names = label_encoder.classes_
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(class_names, [class_counts.get(i, 0) for i in range(len(class_names))])
    ax.set_xlabel("Food Type")
    ax.set_ylabel("Count")
    ax.set_title("Training Data Distribution by Food Class")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# ==========================================
# Page 2: Model Comparison
# ==========================================

elif page == "Model Comparison":
    st.header("Model Performance Comparison")
    
    # Results table
    st.subheader("Accuracy Comparison")
    results_sorted = results_df.sort_values(by="Accuracy", ascending=False)
    st.dataframe(results_sorted, use_container_width=True)
    
    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['green' if i == 0 else 'skyblue' for i in range(len(results_sorted))]
    ax.barh(results_sorted["Model"], results_sorted["Accuracy"], color=colors)
    ax.set_xlabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")
    ax.set_xlim([0.98, 1.0])
    st.pyplot(fig)

# ==========================================
# Page 3: Detailed Analysis
# ==========================================

elif page == "Detailed Analysis":
    st.header("Detailed Model Analysis")
    
    # Model selection
    selected_model = st.selectbox(
        "Select a model to view detailed results:",
        results_df["Model"].tolist()
    )
    
    model = trained_models[selected_model]
    y_pred = model.predict(X_test)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    accuracy = accuracy_score(y_test, y_pred)
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.4f}")
    
    # Confusion Matrix
    st.subheader(f"Confusion Matrix - {selected_model}")
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_, ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    st.pyplot(fig)
    
    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)

# ==========================================
# Page 4: Data Insights
# ==========================================

elif page == "Data Insights":
    st.header("Data Insights")
    
    st.subheader("Dataset Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Shape:**")
        st.write(f"Samples: {X.shape[0]}")
        st.write(f"Features: {X.shape[1]}")
        st.write(f"Classes: {len(label_encoder.classes_)}")
    
    with col2:
        st.write("**Feature Groups:**")
        st.write(f"Numerical: {len(numerical_features)}")
        st.write(f"Categorical: {len(categorical_features)}")
    
    st.subheader("Numerical Features Statistics")
    st.dataframe(X[numerical_features].describe(), use_container_width=True)
    
    st.subheader("Categorical Features")
    for cat_feature in categorical_features:
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{cat_feature}**")
            st.write(X[cat_feature].value_counts())

# ==========================================
# Footer
# ==========================================

st.markdown("---")
st.markdown("""
### About This Dashboard
**NutriClass**: Food Classification using Nutritional Data

This interactive dashboard demonstrates machine learning classification techniques on a food dataset.
The models compare performance on predicting food categories based on nutritional attributes.

**Dataset**: synthetic_food_dataset_imbalanced.csv (31,700+ samples)  
**Models**: 7 different ML algorithms  
**Target**: Food Name (10 classes)
""")

st.caption("Food Classification Dashboard | Machine Learning Project")

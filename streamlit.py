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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier

# ==========================================
# Streamlit Page Config
# ==========================================

st.set_page_config(
    page_title="Food Classification Dashboard",
    layout="wide"
)

st.title("Food Classification using Machine Learning")
st.write("Compare multiple machine learning models on nutritional data")

# ==========================================
# Load Data
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv("data/food_data.csv")

df = load_data()

# ==========================================
# Target and Features
# ==========================================

TARGET_COLUMN = "Food_Name"
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ==========================================
# Feature Groups
# ==========================================

numerical_features = [
    "Calories", "Protein", "Fat", "Carbs", "Sugar", "Fiber",
    "Sodium", "Cholesterol", "Glycemic_Index",
    "Water_Content", "Serving_Size"
]

categorical_features = [
    "Meal_Type", "Preparation_Method",
    "Is_Vegan", "Is_Gluten_Free"
]

# ==========================================
# Preprocessing
# ==========================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
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
    "SVM": SVC(kernel="rbf"),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=len(np.unique(y_encoded)),
        random_state=42
    )
}

# ==========================================
# Train All Models (Cached)
# ==========================================

@st.cache_resource
def train_models():
    trained_models = {}
    metrics = []

    for name, model in models.items():
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

    return trained_models, pd.DataFrame(metrics)

trained_models, results_df = train_models()

# ==========================================
# Results Overview
# ==========================================

st.subheader("Model Performance Comparison")
st.dataframe(results_df.sort_values(by="Accuracy", ascending=False), use_container_width=True)

# ==========================================
# Model Selection
# ==========================================

selected_model = st.selectbox(
    "Select a model to view detailed results:",
    results_df["Model"].tolist()
)

model = trained_models[selected_model]
y_pred = model.predict(X_test)

# ==========================================
# Confusion Matrix
# ==========================================

st.subheader(f"Confusion Matrix – {selected_model}")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# ==========================================
# Classification Report
# ==========================================

st.subheader("Classification Report")
report = classification_report(
    y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
)
st.dataframe(pd.DataFrame(report).transpose())

# ==========================================
# Footer
# ==========================================

st.markdown("---")
st.caption("Food Classification Dashboard | Machine Learning Project")

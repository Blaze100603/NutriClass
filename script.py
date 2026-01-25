# ==========================================
# Food Classification Using Machine Learning
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
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

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier

# ==========================================
# 1. Load Dataset
# ==========================================

data_path = "data/food_data.csv"
df = pd.read_csv(data_path)

print("Dataset Shape:", df.shape)
print(df.head())
print(df.info())

# ==========================================
# 2. Target and Feature Separation
# ==========================================

TARGET_COLUMN = "Food_Name"

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ==========================================
# 3. Feature Classification
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
# 4. Preprocessing Pipelines
# ==========================================

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

# ==========================================
# 5. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ==========================================
# 6. Model Definitions
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="rbf", probability=True),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=len(np.unique(y_encoded)),
        random_state=42
    )
}

# ==========================================
# 7. Training and Evaluation
# ==========================================

results = []

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("pca", PCA(n_components=0.95)),
        ("classifier", model)
    ])

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"{model_name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    results.append({
        "Model": model_name,
        "Accuracy": acc
    })

# ==========================================
# 8. Results Summary
# ==========================================

results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
print("\nModel Performance Comparison:")
print(results_df)

# ==========================================
# 9. Best Model Selection
# ==========================================

best_model_name = results_df.iloc[0]["Model"]
print(f"\nBest Performing Model: {best_model_name}")

# ==========================================
# 10. Feature Importance (Tree-Based Models)
# ==========================================

if best_model_name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
    best_model = models[best_model_name]

    final_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", best_model)
    ])

    final_pipeline.fit(X_train, y_train)

    feature_names = (
        numerical_features +
        list(
            final_pipeline.named_steps["preprocessor"]
            .transformers_[1][1]
            .named_steps["encoder"]
            .get_feature_names_out(categorical_features)
        )
    )

    importances = final_pipeline.named_steps["classifier"].feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df.head(15))
    plt.title("Top 15 Important Features")
    plt.show()

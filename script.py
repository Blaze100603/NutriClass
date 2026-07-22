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
from sklearn.calibration import CalibratedClassifierCV

# ==========================================
# 1. Load Dataset
# ==========================================

data_path = "synthetic_food_dataset_imbalanced.csv"
df = pd.read_csv(data_path)

print("="*60)
print("DATASET LOADING & INSPECTION")
print("="*60)
print(f"\nDataset Shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nData Types:")
print(df.dtypes)
print(f"\nMissing Values:")
print(df.isnull().sum())

# ==========================================
# 2. Data Cleaning
# ==========================================

print("\n" + "="*60)
print("DATA CLEANING")
print("="*60)

# Remove duplicate rows
initial_shape = df.shape
df = df.drop_duplicates()
duplicates_removed = initial_shape[0] - df.shape[0]
print(f"\n✓ Duplicates removed: {duplicates_removed} rows")

# ==========================================
# 3. Target and Feature Separation
# ==========================================

TARGET_COLUMN = "Food_Name"

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\nTarget column: {TARGET_COLUMN}")
print(f"Classes: {label_encoder.classes_}")
print(f"Target distribution:\n{pd.Series(y_encoded).value_counts().sort_index()}")

# ==========================================
# 4. Feature Classification
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

print(f"\nNumerical features ({len(numerical_features)}): {numerical_features}")
print(f"\nCategorical features ({len(categorical_features)}): {categorical_features}")

# ==========================================
# 5. Outlier Detection and Handling (IQR Method)
# ==========================================

print("\n" + "="*60)
print("OUTLIER DETECTION & REMOVAL (IQR Method)")
print("="*60)

def remove_outliers_class_aware(data, target, features):
    """Remove outliers using Class-Aware IQR method
    
    Instead of global outlier removal which can eliminate entire food classes,
    this method removes outliers WITHIN each food class.
    """
    all_indices_to_keep = []
    total_outliers = 0
    
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
        
        class_outliers = (~class_outlier_mask).sum()
        samples_kept = class_outlier_mask.sum()
        original_count = class_mask.sum()
        
        print(f"  {food_class:12} | Original: {original_count:5} | Kept: {samples_kept:5} | Outliers: {class_outliers:5} | Retention: {samples_kept/original_count*100:5.1f}%")
        
        all_indices_to_keep.extend(data_class[class_outlier_mask].index)
        total_outliers += class_outliers
    
    final_mask = data.index.isin(all_indices_to_keep)
    return data[final_mask], final_mask, total_outliers

X_before = X.shape[0]
X, outlier_mask, outliers_removed = remove_outliers_class_aware(X, y, numerical_features)
y = y[outlier_mask]
y_encoded_filtered = label_encoder.fit_transform(y)

print(f"\n✓ Total outliers removed: {outliers_removed} rows")
print(f"✓ Dataset shape after cleaning: {X.shape}")
print(f"✓ Data retention: {X.shape[0] / X_before * 100:.1f}%")
print(f"✓ All {len(y.unique())} food classes preserved")

# ==========================================
# 6. Preprocessing Pipelines
# ==========================================

print("\n" + "="*60)
print("PREPROCESSING PIPELINE CONFIGURATION")
print("="*60)

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

print("\n✓ Numeric Pipeline: Imputation (median) → Standardization")
print("✓ Categorical Pipeline: Imputation (most_frequent) → One-Hot Encoding")
print("✓ ColumnTransformer configured successfully")

# ==========================================
# 7. Train-Test Split
# ==========================================

print("\n" + "="*60)
print("TRAIN-TEST SPLIT (80-20)")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded_filtered,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded_filtered
)

print(f"\n✓ Training set size: {X_train.shape[0]} samples")
print(f"✓ Test set size: {X_test.shape[0]} samples")
print(f"✓ Feature count: {X_train.shape[1]}")

# ==========================================
# 8. Model Definitions
# ==========================================

print("\n" + "="*60)
print("MODEL DEFINITIONS")
print("="*60)

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
        num_class=len(np.unique(y_encoded_filtered)),
        random_state=42,
        verbosity=0
    )
}

print(f"\n✓ {len(models)} models configured")
for model_name in models.keys():
    print(f"  - {model_name}")

# ==========================================
# 9. Training and Evaluation
# ==========================================

print("\n" + "="*60)
print("MODEL TRAINING & EVALUATION")
print("="*60)

results = []

for model_name, model in models.items():
    print(f"\n>> Training {model_name}...")

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("pca", PCA(n_components=0.95)),
        ("classifier", model)
    ])

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"   ✓ {model_name} Accuracy: {acc:.4f}")
    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png")
    plt.show()

    results.append({
        "Model": model_name,
        "Accuracy": acc
    })

# ==========================================
# 10. Results Summary
# ==========================================

print("\n" + "="*60)
print("MODEL PERFORMANCE COMPARISON")
print("="*60)

results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
print("\n", results_df.to_string(index=False))

# ==========================================
# 11. Best Model Selection
# ==========================================

best_model_name = results_df.iloc[0]["Model"]
print(f"\n✓ Best Performing Model: {best_model_name}")
print(f"  Accuracy: {results_df.iloc[0]['Accuracy']:.4f}")

# ==========================================
# 12. Feature Importance (Tree-Based Models)
# ==========================================

print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

if best_model_name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
    print(f"\nExtracting feature importance from {best_model_name}...")
    
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

    print(f"\n✓ Top 15 Important Features:")
    print(importance_df.head(15).to_string(index=False))

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df.head(15))
    plt.title(f"Top 15 Important Features - {best_model_name}")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()
else:
    print(f"\nNote: {best_model_name} is not a tree-based model.")
    print("Feature importance only available for tree-based models (Random Forest, Gradient Boosting, XGBoost).")

print("\n" + "="*60)
print("✓ ANALYSIS COMPLETE")
print("="*60)

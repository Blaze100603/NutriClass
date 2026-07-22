# ==========================================
# Improved Preprocessing - Class-Aware Outlier Removal
# ==========================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("synthetic_food_dataset_imbalanced.csv")

print("="*70)
print("IMPROVED PREPROCESSING - CLASS-AWARE OUTLIER REMOVAL")
print("="*70)

print(f"\nOriginal Dataset: {df.shape[0]} samples, {df['Food_Name'].nunique()} classes")

# Features
numerical_features = [
    "Calories", "Protein", "Fat", "Carbs", "Sugar", "Fiber",
    "Sodium", "Cholesterol", "Glycemic_Index",
    "Water_Content", "Serving_Size"
]

# Remove duplicates
df = df.drop_duplicates()
print(f"\nAfter duplicate removal: {df.shape[0]} samples")

# ==========================================
# CLASS-AWARE OUTLIER REMOVAL
# ==========================================

print(f"\n{'='*70}")
print("CLASS-AWARE OUTLIER REMOVAL (IQR by Food Class)")
print(f"{'='*70}\n")

X = df.drop(columns=["Food_Name"])
y = df["Food_Name"]

all_indices_to_keep = []
outlier_stats = []

for food_class in sorted(y.unique()):
    class_mask = y == food_class
    X_class = X[class_mask]
    
    class_outliers = 0
    class_mask_final = pd.Series([True] * len(X_class), index=X_class.index)
    
    for feature in numerical_features:
        Q1 = X_class[feature].quantile(0.25)
        Q3 = X_class[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        feature_mask = (X_class[feature] >= lower_bound) & (X_class[feature] <= upper_bound)
        class_mask_final = class_mask_final & feature_mask
    
    class_outliers = (~class_mask_final).sum()
    samples_kept = class_mask_final.sum()
    original_count = class_mask.sum()
    
    print(f"{food_class:12} | Original: {original_count:5} | Kept: {samples_kept:5} | Outliers: {class_outliers:5} | Retention: {samples_kept/original_count*100:5.1f}%")
    
    all_indices_to_keep.extend(X_class[class_mask_final].index)
    outlier_stats.append({
        "Food": food_class,
        "Original": original_count,
        "Kept": samples_kept,
        "Outliers": class_outliers,
        "Retention %": samples_kept / original_count * 100
    })

# Apply filtering
X_cleaned = X.loc[all_indices_to_keep]
y_cleaned = y.loc[all_indices_to_keep]

print(f"\n{'='*70}")
print("RESULTS")
print(f"{'='*70}")

print(f"\nOriginal Samples: {len(df)}")
print(f"Cleaned Samples: {len(X_cleaned)}")
print(f"Outliers Removed: {len(df) - len(X_cleaned)}")
print(f"Data Retention: {len(X_cleaned) / len(df) * 100:.1f}%")

print(f"\nClasses Preserved: {y_cleaned.nunique()} / {y.nunique()}")
print(f"Classes: {sorted(y_cleaned.unique())}")

# Statistics
stats_df = pd.DataFrame(outlier_stats)
print(f"\n{'='*70}")
print("OUTLIER REMOVAL STATISTICS BY CLASS")
print(f"{'='*70}")
print(stats_df.to_string(index=False))

print(f"\n{'='*70}")
print("COMPARISON: Global vs Class-Aware Outlier Removal")
print(f"{'='*70}")

# Simulate global removal for comparison
X_global = df.drop(columns=["Food_Name"])
y_global = df["Food_Name"]
mask_global = pd.Series([True] * len(X_global), index=X_global.index)

for feature in numerical_features:
    Q1 = X_global[feature].quantile(0.25)
    Q3 = X_global[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    feature_mask = (X_global[feature] >= lower_bound) & (X_global[feature] <= upper_bound)
    mask_global = mask_global & feature_mask

X_global_cleaned = X_global[mask_global]
y_global_cleaned = y_global[mask_global]

print(f"\nGLOBAL OUTLIER REMOVAL:")
print(f"  Samples: {len(df)} → {len(X_global_cleaned)}")
print(f"  Classes: {y.nunique()} → {y_global_cleaned.nunique()}")
print(f"  Lost Classes: {set(y.unique()) - set(y_global_cleaned.unique())}")

print(f"\nCLASS-AWARE OUTLIER REMOVAL:")
print(f"  Samples: {len(df)} → {len(X_cleaned)}")
print(f"  Classes: {y.nunique()} → {y_cleaned.nunique()}")
print(f"  Lost Classes: {set(y.unique()) - set(y_cleaned.unique())}")

print(f"\n{'='*70}")
print("✓ RECOMMENDATION: Use CLASS-AWARE outlier removal")
print("  - Preserves all food classes")
print("  - Removes only true outliers within each class")
print("  - Prevents bias from removing nutritionally-distinct foods")
print(f"{'='*70}")

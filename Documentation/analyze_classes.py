import pandas as pd

# Load raw data
df_raw = pd.read_csv('synthetic_food_dataset_imbalanced.csv')
print('RAW DATASET - Classes:')
print(df_raw['Food_Name'].value_counts().sort_index())
print(f'\nTotal unique classes: {df_raw["Food_Name"].nunique()}')

# Simulate outlier removal
numerical_features = [
    'Calories', 'Protein', 'Fat', 'Carbs', 'Sugar', 'Fiber',
    'Sodium', 'Cholesterol', 'Glycemic_Index',
    'Water_Content', 'Serving_Size'
]

X = df_raw.drop(columns=['Food_Name'])
y = df_raw['Food_Name']

mask = pd.Series([True] * len(X), index=X.index)
for feature in numerical_features:
    Q1 = X[feature].quantile(0.25)
    Q3 = X[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    feature_mask = (X[feature] >= lower_bound) & (X[feature] <= upper_bound)
    mask = mask & feature_mask

X_cleaned = X[mask]
y_cleaned = y[mask]

print(f'\n\nCLEANED DATASET - Classes:')
print(y_cleaned.value_counts().sort_index())
print(f'\nTotal unique classes: {y_cleaned.nunique()}')

print(f'\n\nCLASSES REMOVED:')
raw_classes = set(df_raw['Food_Name'].unique())
cleaned_classes = set(y_cleaned.unique())
removed = raw_classes - cleaned_classes
print(f'Removed: {removed}')
print(f'Count: {len(removed)} classes')

print(f'\n\nSAMPLES PER CLASS - BEFORE vs AFTER:')
print(f'{"Food":12} | {"Raw":>5} | {"Cleaned":>7} | {"Removed":>7} | {"% Removed":>9}')
print('-' * 60)
for food in sorted(df_raw['Food_Name'].unique()):
    raw_count = (df_raw['Food_Name'] == food).sum()
    cleaned_count = (y_cleaned == food).sum() if food in cleaned_classes else 0
    removed_count = raw_count - cleaned_count
    pct = removed_count/raw_count*100 if raw_count > 0 else 0
    print(f'{food:12} | {raw_count:5} | {cleaned_count:7} | {removed_count:7} | {pct:8.1f}%')

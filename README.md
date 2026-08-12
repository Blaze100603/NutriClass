NutriClass — Food Classification Using Machine Learning

Project Overview

NutriClass is an end-to-end machine learning project that classifies food items using nutritional, dietary, and preparation attributes.

The project evaluates multiple machine learning approaches for a 10-class food classification problem and provides an interactive Streamlit dashboard for comparing model performance, inspecting classification errors, exploring the dataset, and understanding which input features contribute most to predictions.

The workflow covers the complete machine learning lifecycle:

Raw Data → Data Cleaning → Train/Test Split → Preprocessing → Model Training → Model Evaluation → Explainability → Interactive Dashboard

Important: The dataset used by this project is synthetic. Model performance should therefore be interpreted as an evaluation of the classification pipeline rather than evidence of real-world food-classification accuracy.

Problem Statement

Given nutritional, dietary, and preparation attributes for a food item, the objective is to predict its food type.

The system is designed to:

Classify food items into their respective categories

Compare multiple traditional and ensemble machine learning models

Evaluate performance using metrics appropriate for an imbalanced multi-class problem

Identify which nutritional and categorical features contribute most to predictions

Provide an interactive interface for exploring model performance and dataset characteristics

Business & Analytical Questions

The project is designed around questions such as:

Which machine learning model performs best at distinguishing food types?

How consistent is model performance across different food classes?

Which nutritional characteristics are most useful for classification?

Does class imbalance materially affect model evaluation?

Which food classes are most frequently confused with one another?

How do nutritional and preparation attributes differ across food types?

Potential applications include:

Dietary tracking systems

Food-management applications

Nutritional analytics

Menu and food-service analysis

Recommendation systems

Automated food categorization

These are potential applications; the current project focuses on the classification pipeline and analytical dashboard.

Dataset

The dataset is stored in:

synthetic_food_dataset_imbalanced.csv

It contains 31,700+ observations and 10 food classes.

Numerical Features

Calories

Protein

Fat

Carbs

Sugar

Fiber

Sodium

Cholesterol

Glycemic_Index

Water_Content

Serving_Size

Categorical Features

Meal_Type

Preparation_Method

Is_Vegan

Is_Gluten_Free

Target Variable

Food_Name — food type/classification target

The dataset is intentionally treated as an imbalanced multi-class classification problem, making metrics such as Macro F1 important when comparing models.

Machine Learning Pipeline

Raw Dataset
     ↓
Duplicate Removal
     ↓
Train/Test Split
     ↓
Training-Only Outlier Treatment
     ↓
Missing Value Imputation
     ↓
Categorical Encoding
     ↓
Feature Scaling
     ↓
Optional PCA
     ↓
Model Training
     ↓
Multi-Class Evaluation
     ↓
Feature Importance
     ↓
Streamlit Dashboard

Data Cleaning

Exact duplicate records are removed before modeling.

Class-aware IQR outlier treatment is then applied to the training data only. The test set remains untouched so that final evaluation is performed on unseen observations.

This prevents test-set information from influencing the outlier-treatment step.

Preprocessing

Numerical features use:

Median imputation

Standardization using StandardScaler

Categorical features use:

Most-frequent imputation

One-hot encoding

Unknown-category handling

All preprocessing is implemented through scikit-learn pipelines and a ColumnTransformer.

Dimensionality Reduction

PCA retaining 95% of explained variance is used for models that can benefit from dimensionality reduction:

Logistic Regression

KNN

SVM

PCA is intentionally not applied to tree-based models, allowing their original feature structure to remain more interpretable.

Models Evaluated

The project compares seven classification algorithms:

Logistic Regression

Decision Tree

Random Forest

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Gradient Boosting

XGBoost

The objective is not simply to identify the most complex model, but to compare different model families using a consistent evaluation framework.

Model Evaluation

Because the dataset is imbalanced, model performance is evaluated using multiple metrics:

Accuracy

Macro Precision

Macro Recall

Macro F1

Weighted F1

Confusion Matrix

Per-class Precision, Recall, and F1

Why Macro F1?

Accuracy can hide poor performance on minority classes.

Macro F1 calculates F1 independently for each class and then gives every class equal weight. This makes it a more useful primary comparison metric when the class distribution is imbalanced.

The dashboard therefore ranks models primarily by Macro F1, while still reporting the other metrics.

Dashboard

The Streamlit dashboard contains five analytical sections.

1. Overview

Provides:

Dataset size

Training/test sample counts

Number of classes

Number of features

Food-class distribution

Classification objective

2. Model Comparison

Provides:

Accuracy comparison

Macro Precision

Macro Recall

Macro F1

Weighted F1

PCA usage by model

Best-performing model based on Macro F1

3. Detailed Analysis

Allows users to select an individual model and examine:

Accuracy

Macro Precision

Macro Recall

Macro F1

Confusion Matrix

Full Classification Report

4. Feature Importance

Uses permutation importance evaluated on the untouched test set to identify which original input features have the greatest influence on model performance.

This provides a model-agnostic approach to interpreting the relationship between nutritional/dietary characteristics and predicted food types.

5. Data Insights

Provides:

Dataset dimensions

Feature-group breakdown

Numerical feature statistics

Categorical feature distributions

Training-data characteristics

Project Structure

food-classification/
│
├── synthetic_food_dataset_imbalanced.csv
│
├── dashboards/
│   └── streamlit_app.py
│
├── README.md
│
└── requirements.txt

If the project is organized differently in the repository, update this section to match the final repository structure.

Technology Stack

Technology

Purpose

Python

Machine learning and data processing

Pandas

Data manipulation and analysis

NumPy

Numerical computation

Scikit-learn

Preprocessing, modeling, evaluation, and pipelines

XGBoost

Gradient-boosted classification

Matplotlib

Data visualization

Seaborn

Statistical visualization

Streamlit

Interactive dashboard

Quick Start

1. Clone the Repository

git clone <repository-url>
cd <repository-folder>

2. Install Dependencies

A virtual environment is recommended.

python -m pip install -r requirements.txt

3. Run the Streamlit Dashboard

streamlit run dashboards/streamlit_app.py

The dashboard will open in your browser.

Methodological Considerations

Synthetic Dataset

The dataset is synthetic, so extremely high model performance should not be interpreted as evidence that the same performance would occur on real-world food data.

The primary purpose of the project is to demonstrate:

End-to-end ML workflow

Multi-class classification

Model comparison

Appropriate evaluation for imbalanced data

Explainability

Interactive analytical reporting

Test-Set Integrity

The train/test split is performed before training-dependent outlier treatment.

The test dataset is kept untouched for final model evaluation.

Model-Specific Preprocessing

PCA is applied selectively rather than universally.

Linear, distance-based, and kernel-based models use PCA, while tree-based models operate without PCA so that their relationship with the original feature space remains easier to interpret.

Key Takeaways

This project demonstrates practical experience with:

Data cleaning and preprocessing

Handling mixed numerical and categorical data

Train/test methodology

Class-aware outlier treatment

Feature scaling and encoding

Dimensionality reduction

Multi-class classification

Ensemble learning

Imbalanced-data evaluation

Model comparison

Confusion-matrix analysis

Feature importance and explainability

Interactive dashboard development

The project emphasizes the complete analytical workflow rather than treating model training as the final objective.

Future Improvements

Add cross-validation for more robust model comparison

Add hyperparameter tuning

Add ROC-AUC / Precision-Recall analysis where appropriate

Add SHAP-based explainability for the best-performing model

Add automated experiment tracking

Add model persistence and loading instead of retraining on dashboard startup

Add a prediction interface for new food observations

Add automated tests

Add CI/CD

Evaluate the pipeline on a real-world food/nutrition dataset

Compare synthetic-data performance against real-world generalization

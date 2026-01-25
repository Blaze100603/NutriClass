# Food Classification Using Machine Learning

## Overview

In the era of increasing dietary awareness, accurately classifying food items based on their nutritional attributes is critical for building intelligent food and diet management systems.  
This project implements an **end-to-end machine learning pipeline** that classifies food items using tabular nutritional data such as calories, protein, fat, carbohydrates, sugar, and preparation attributes.

Multiple traditional and ensemble-based machine learning models are trained, evaluated, and compared.  
Additionally, an **interactive Streamlit dashboard** is provided to visualize and compare all models simultaneously.

---

## Problem Statement

Given a dataset containing nutritional and dietary attributes of food items, the objective is to develop a **multi-class classification system** that can:

- Accurately classify food items into their respective categories
- Compare the performance of multiple machine learning models
- Provide insights into which nutritional features distinguish different food types

---

## Dataset Description

The dataset is a CSV file with the following columns:

### Numerical Features
- Calories  
- Protein  
- Fat  
- Carbs  
- Sugar  
- Fiber  
- Sodium  
- Cholesterol  
- Glycemic_Index  
- Water_Content  
- Serving_Size  

### Categorical Features
- Meal_Type  
- Preparation_Method  
- Is_Vegan  
- Is_Gluten_Free  

### Target Variable
- Food_Name  

---

## Project Structure



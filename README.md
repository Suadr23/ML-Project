# ML-Project
## Project Overview

This project performs customer segmentation for a mall dataset using machine learning classification techniques.

The main goal is to classify customers based on their spending score into two categories:  
- High spender (Spending Score >= 50)  
- Low spender (Spending Score < 50)

The dataset contains features such as Gender, Age, and Annual Income (k$).

### Models Used:
- Logistic Regression with hyperparameter tuning using GridSearchCV
- AdaBoost Classifier with hyperparameter tuning

### Workflow:
1. Load and explore the dataset.
2. Visualize distributions of key features (Age, Income, Spending Score).
3. Encode categorical variables (Gender).
4. Split data into train and test sets.
5. Scale features using StandardScaler.
6. Train models with hyperparameter tuning.
7. Evaluate models using accuracy, precision, recall, and F1-score.
8. Visualize performance comparison between models.



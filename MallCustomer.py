import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv(r"C:\Users\USER PC\OneDrive\Desktop\ML Project Mall customer\Mall Customers.csv")

print("First 5 rows:")
print(df.head())

print("\n Info:")
print(df.info())

print("\n Describe:")
print(df.describe())

print("\n Missing values:")
print(df.isnull().sum())

plt.figure(figsize=(18, 4))

plt.subplot(1, 3, 1)
plt.hist(df['Age'], bins=10, edgecolor='black')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Number of Customers')

plt.subplot(1, 3, 2)
plt.hist(df['Annual Income (k$)'], bins=10, edgecolor='black', color='green')
plt.title('Distribution of Annual Income')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Number of Customers')

plt.subplot(1, 3, 3)
plt.hist(df['Spending Score (1-100)'], bins=10, edgecolor='black', color='orange')
plt.title('Distribution of Spending Score')
plt.xlabel('Spending Score')
plt.ylabel('Number of Customers')

plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender')
plt.title('Income vs Spending Score')
plt.show()

df['Gender'] = df['Gender'].replace({'Male': 0, 'Female': 1})
df['Spending_Label'] = df['Spending Score (1-100)'].apply(lambda x: 1 if x >= 50 else 0)

X = df[['Gender', 'Age', 'Annual Income (k$)']]
y = df['Spending_Label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("First 5 rows of X :")
print(X_scaled[:5])

print("\nFirst 5 labels (y):")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model_logreg = LogisticRegression()
pipe_logreg = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', model_logreg)
])

param_grid_logreg = {
    'clf__C': [0.1, 1, 10],
    'clf__max_iter': [100, 200, 300]
}

grid_logreg = GridSearchCV(pipe_logreg, param_grid_logreg, cv=5, scoring='accuracy')
grid_logreg.fit(X_train, y_train)

print("LogisticRegression best params:", grid_logreg.best_params_)
print("LogisticRegression best CV accuracy:", grid_logreg.best_score_)

best_logreg = grid_logreg.best_estimator_
y_pred_logreg = best_logreg.predict(X_test)
print("LogisticRegression test accuracy:", accuracy_score(y_test, y_pred_logreg))

pipe_ada = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', AdaBoostClassifier())
])

param_grid_ada = {
    'clf__n_estimators': [50, 100, 150],
    'clf__learning_rate': [0.01, 0.1, 1]
}

grid_ada = GridSearchCV(pipe_ada, param_grid_ada, cv=5, scoring='accuracy')
grid_ada.fit(X_train, y_train)

print("\nAdaBoost best params:", grid_ada.best_params_)
print("AdaBoost best CV accuracy:", grid_ada.best_score_)

best_ada = grid_ada.best_estimator_
y_pred_ada = best_ada.predict(X_test)
print("AdaBoost test accuracy:", accuracy_score(y_test, y_pred_ada))

X_resampled, y_resampled = resample(X_train, y_train, n_samples=100, random_state=42)
print("\nResampled data size:", X_resampled.shape)

accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
precision_logreg = precision_score(y_test, y_pred_logreg)
recall_logreg = recall_score(y_test, y_pred_logreg)
f1_logreg = f1_score(y_test, y_pred_logreg)

accuracy_ada = accuracy_score(y_test, y_pred_ada)
precision_ada = precision_score(y_test, y_pred_ada)
recall_ada = recall_score(y_test, y_pred_ada)
f1_ada = f1_score(y_test, y_pred_ada)

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
logreg_scores = [accuracy_logreg, precision_logreg, recall_logreg, f1_logreg]
ada_scores = [accuracy_ada, precision_ada, recall_ada, f1_ada]

x = range(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar([p - width/2 for p in x], logreg_scores, width=width, label='Logistic Regression', color='skyblue')
plt.bar([p + width/2 for p in x], ada_scores, width=width, label='AdaBoost', color='salmon')
plt.xticks(x, metrics)
plt.ylim(0, 1)
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.legend()

for i, (log, ada) in enumerate(zip(logreg_scores, ada_scores)):
    plt.text(i - width/2, log + 0.02, f'{log:.2f}', ha='center')
    plt.text(i + width/2, ada + 0.02, f'{ada:.2f}', ha='center')

plt.tight_layout()
plt.show()

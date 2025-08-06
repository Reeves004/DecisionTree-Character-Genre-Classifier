"""
Fictional Character Genre Classifier
Dataset: fictional_characters.xlsx
Algorithm: Decision Tree Classifier
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# --- Data Loading ---
try:
    data = pd.read_excel('fictional_characters.xlsx', engine='openpyxl')
    print("Dataset loaded successfully!\n")
except Exception as e:
    print(f"Error: {e}")
    exit()

# --- Target Selection ---
TARGET = 'Genre'  # Change if needed
print(f"Target variable: {TARGET}\n")

# --- Preprocessing ---
# Encode categorical target
le = LabelEncoder()
y = le.fit_transform(data[TARGET])

# Encode features (simplified approach)
X = data.drop(TARGET, axis=1)
for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}\n")

# --- Decision Tree Model ---
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,  # Restricted for interpretability
    random_state=42
)
model.fit(X_train, y_train)

# --- Evaluation ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}\n")

print("Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=le.classes_,
    zero_division=0  # Suppresses warnings
))

# --- Visualization ---
plt.figure(figsize=(25, 15), dpi=300)  # Double the size and resolution
plot_tree(
    model,
    feature_names=X.columns,
    class_names=le.classes_,
    filled=True,
    rounded=True,
    fontsize=10,  # Larger font
    proportion=True,  # Shows percentages
    impurity=False  # Cleaner look
)
plt.tight_layout()  # Prevents label cutoff
plt.savefig('high_res_tree.png', bbox_inches='tight', dpi=300)  # 300 DPI for print quality
print("High-resolution tree saved as 'high_res_tree.png'")

# --- Save Model ---
import joblib
joblib.dump(model, 'genre_classifier.pkl')
print("Model saved as 'genre_classifier.pkl'")

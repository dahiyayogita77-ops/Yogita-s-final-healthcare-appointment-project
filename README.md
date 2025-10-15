# ------------------------------------------------------------
# Healthcare Appointment No-Show Prediction
# ------------------------------------------------------------
# Author: Yogita
# Dataset: healthcare_no_show.xlsx
# ------------------------------------------------------------

# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Step 2: Load dataset
file_path = "C:\\Users\\dell\\OneDrive\\Documents\\Yogita's final (no show prediction ) project.xlsx"
df = pd.read_excel(file_path)

print("✅ Data Loaded Successfully!\n")
print(df.head())

# Step 3: Encode categorical variables
le = LabelEncoder()
for col in ['Gender', 'Diabetes', 'Hypertension', 'AppointmentDay', 'NoShow']:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])

print("\n✅ Encoded Data Sample:")
print(df.head())

# Step 4: Check target distribution
print("\n📊 NoShow Value Counts:")
print(df['NoShow'].value_counts())

# Step 5: Split data into features (X) and target (y)
X = df.drop(['Patient_ID', 'NoShow'], axis=1)
y = df['NoShow']

# Step 6: Train-test split (keeps balance between Yes/No)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Step 7: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Make predictions
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
print("\n✅ Model Evaluation Results:")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 10: Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.colorbar()
plt.show()

# Step 11: Feature Importance
importance = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(6, 4))
importance.sort_values().plot(kind='barh', color='skyblue')
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()

# Step 12: Visualize Accuracy and Class Distribution
accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)

# Accuracy Bar Chart
plt.figure(figsize=(4, 4))
plt.bar(['Model Accuracy'], [accuracy], color='lightgreen')
plt.title(f'Model Accuracy: {accuracy}%')
plt.ylim(0, 100)
plt.show()

# No-Show Distribution Pie Chart
plt.figure(figsize=(5, 5))
df['NoShow'].value_counts().plot.pie(
    autopct='%1.1f%%', startangle=90, colors=['lightblue', 'pink'], labels=['Show', 'No-Show']
)
plt.title('Appointment Attendance Distribution')
plt.ylabel('')
plt.show()

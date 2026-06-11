import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("data/heart.csv")

# Feature dan Target
X = df.drop("target", axis=1)
y = df["target"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Training
rf.fit(X_train, y_train)

# Evaluasi
y_pred = rf.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"Akurasi Model : {accuracy*100:.2f}%")

# Simpan Model
joblib.dump(
    rf,
    "model/random_forest.pkl"
)

print("Model berhasil disimpan!")

import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# =========================
# DATA (عدّل حسب الداتا بتاعتك)
# =========================
# لازم يكون عندك CSV فيه عمودين:
# text , emotion

df = pd.read_csv("data.csv")

X = df["text"]
y = df["emotion"]

# =========================
# VECTORIZER
# =========================
vectorizer = TfidfVectorizer(max_features=3480)
X_vec = vectorizer.fit_transform(X)

# =========================
# MODEL
# =========================
model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

# =========================
# SAVE
# =========================
joblib.dump(vectorizer, "emotion-models/vectorizer.pkl")
joblib.dump(model, "emotion-models/text_model.pkl")

print("DONE ✅")
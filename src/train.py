from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from utils import load_data, save_model
import joblib


# =========================
# 1. LOAD DATA
# =========================
df = load_data("C:\\Users\\dhruv\\OneDrive\\Desktop\\Project Ticket\\Data\\cleaned_data.csv")

print("Columns in dataset:", df.columns)


# =========================
# 2. FEATURES & LABELS
# =========================
# Using already cleaned text + encoded labels
X = df['text_clean']
y = df['type_encoded']


# =========================
# 3. VECTORIZATION
# =========================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)


# =========================
# 4. TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# 5. TRAIN MODEL
# =========================
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


# =========================
# 6. SAVE LABEL MAPPING (NEW 🔥)
# =========================
# Maps encoded numbers → actual category names
label_mapping = dict(enumerate(df['type'].astype('category').cat.categories))
joblib.dump(label_mapping, "../models/label_mapping.pkl")


# =========================
# 7. SAVE MODEL + VECTORIZER
# =========================
save_model(model, "../models/best_model.pkl")
save_model(vectorizer, "../models/vectorizer.pkl")


# =========================
# 8. DONE
# =========================
print("✅ Model + vectorizer + label mapping saved successfully!")
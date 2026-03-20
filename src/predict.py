from utils import load_model

model = load_model("../models/best_model.pkl")
vectorizer = load_model("../models/vectorizer.pkl")
label_mapping = load_model("../models/label_mapping.pkl")


def predict_ticket(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return label_mapping[prediction[0]]


if __name__ == "__main__":
    sample = "My internet is not working"
    print("Predicted category:", predict_ticket(sample))
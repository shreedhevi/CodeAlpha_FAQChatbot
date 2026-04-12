from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# FAQ Data
questions = [
    "what is your name",
    "how are you",
    "what is ai",
    "what is python",
    "who created you",
    "what is machine learning",
    "what is chatbot"
]

answers = [
    "I am FAQ Chatbot.",
    "I am doing great!",
    "AI means Artificial Intelligence.",
    "Python is a programming language.",
    "I was created by ShreeDhevi.",
    "Machine Learning is a part of AI that learns from data.",
    "A chatbot is a program that talks with users."
]

# NLP Model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

def get_response(user_input):
    user_vec = vectorizer.transform([user_input.lower()])
    similarity = cosine_similarity(user_vec, X)
    index = similarity.argmax()

    # No match condition
    if similarity[0][index] < 0.3:
        return "Sorry, I don't understand your question."

    return answers[index]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    response = get_response(user_input)
    return jsonify({"reply": response})

if __name__ == "__main__":
    app.run(debug=True)
    
from flask import Flask, request, jsonify
from flask_cors import CORS
from utils import predict_class, get_response

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Welcome to the Chatbot API!"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    msg = data.get("message")
    intents = predict_class(msg)
    res = get_response(intents)
    return jsonify({"response": res})

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
from utils import predict_class, get_response
import os

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return "GoJerowaru Chatbot API - Ready!"

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()

        if not data or "message" not in data:
            return jsonify({"error": "Field 'message' diperlukan"}), 400

        message = data["message"]
        context = data.get("context", None)

        intents = predict_class(message)
        response, new_context = get_response(intents)

        return jsonify({
            "response": response,
            "context": new_context
        })

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)

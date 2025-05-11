from utils import predict_class, get_response

def chat():
    print("Bot siap! Ketik 'quit' untuk keluar.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Bot: Sampai jumpa!")
            break
        intents = predict_class(user_input)
        response = get_response(intents)
        print("Bot:", response)

if __name__ == "__main__":
    chat()

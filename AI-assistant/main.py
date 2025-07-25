from AI.gemini import ask_model
from database.init_data import load_history, exists_history
def main():
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the AI assistant. Goodbye!")
            break
        response = ask_model(user_input)
        print(f"Assistant: {response}")
    
if __name__ == "__main__":
    main()
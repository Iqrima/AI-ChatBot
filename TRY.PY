from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Access the GEMINI_API_KEY
api_key = os.getenv("GEMINI_API_KEY")

# Raise an error if the API key is not found
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

# Configure the Gemini API
genai.configure(api_key=api_key)

# Define chatbot generation configuration
generation_config = {
    "temperature": 0.7,  # Controls creativity (0.0: deterministic, 1.0: more creative)
    "top_p": 0.9,       # Controls the likelihood of sampling less probable tokens
    "top_k": 40,        # Limits the sampling to the top-k tokens
    "max_output_tokens": 512,  # Maximum tokens in the response
    "response_mime_type": "text/plain",
}

# Initialize the Generative Model
try:
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        generation_config=generation_config,
    )
    print("Gemini model initialized successfully.")
except Exception as e:
    raise ValueError(f"Failed to initialize Gemini model: {e}")

# Start a chat session
chat_session = model.start_chat(history=[])


def chatbot():
    """
    Chatbot main loop that interacts with the user.
    """
    print("Chatbot is ready! Type 'exit' to quit.")
    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        try:
            # Send user input to the Gemini API and get a response
            response = chat_session.send_message(user_input)
            if response and hasattr(response, "text"):
                print(f"Chatbot: {response.text}")
            else:
                print("Chatbot: Sorry, I couldn't understand your request.")
        except Exception as e:
            print(f"Chatbot: An error occurred: {e}")


# Run the chatbot
if __name__ == "__main__":
    chatbot()

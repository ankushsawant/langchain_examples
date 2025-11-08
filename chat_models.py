from http.client import responses
from dotenv import load_dotenv
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Load environment variables from .env
load_dotenv()

def genai_test():
    print("Hello Google GenAI World i.e. w/o Langchain")

    # The client gets the API key from the environment variable `GOOGLE_API_KEY`.
    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Who is Barack Obama?"
    )

    print(response.text)

def chat_model_basic():
    print("Chat Model Basic")

    # Create a ChatGoogleGenAI model
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    result = model.invoke("Who is Narendra Modi?")
    print(f"result: {result}")
    print(f"content: {result.content}")

def chat_model_basic_conversation():
    print("Chat Model Basic Conversation")

    # Create a ChatGoogleGenAI model
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    # SystemMessage:
    #   Message for priming AI behavior, usually passed in as the first of a sequenc of input messages.
    # HumanMessagse:
    #   Message from a human to the AI model.
    messages = [
        SystemMessage(content="Solve the following math problems"),
        HumanMessage(content="What is 81 divided by 9?"),
    ]

    # Invoke the model with messages
    result = model.invoke(messages)
    print(f"Answer from AI: {result.content}")

    # AIMessage:
    #   Message from an AI.
    messages = [
        SystemMessage(content="Solve the following math problems"),
        HumanMessage(content="What is 81 divided by 9?"),
        AIMessage(content="81 divided by 9 is 9."),
        HumanMessage(content="What is 10 times 5?"),
    ]

    # Invoke the model with messages
    result = model.invoke(messages)
    print(f"Answer from AI: {result.content}")

def chat_model_conversation_with_user():
    print("Chat Model Basic Conversation with User")

    # Create a ChatGoogleGenAI model
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    # Create a list to store messages
    chat_history = []
    message = SystemMessage(content="You are a helpful AI assistant")
    chat_history.append(message) # Add (optional) system message

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        chat_history.append(HumanMessage(content=query)) # Add user message

        # call the model with entire chat history
        response = model.invoke(chat_history)
        chat_history.append(AIMessage(content=response.content)) # Add AI message

        print(response.content)

    print(f"Chat history:{chat_history}")


if __name__ == "__main__":
    genai_test()
    # chat_model_basic()
    # chat_model_basic_conversation()
    # chat_model_conversation_with_user()
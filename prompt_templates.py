from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env
load_dotenv()

def prompt_templates():
    print("Prompt templates")

    # Create a ChatGoogleGenAI model
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    # Part 1
    # Joke prompt template with multiple placeholders or variables
    multi_inputs = "Tell us {number_of_jokes} funny jokes about {subject}"

    joke_template = ChatPromptTemplate.from_template(multi_inputs)
    joke_prompt = joke_template.invoke({"number_of_jokes": "2", "subject": "doctors"})
    # print(joke_prompt)

    result = model.invoke(joke_prompt)
    print(result.content)

    # Part 2
    # Joke prompt template with System and Human Messages (Using Tuples)
    joke_messages = [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
    joke_template_messages = ChatPromptTemplate.from_messages(joke_messages)
    joke_prompt_messages = joke_template_messages.invoke({"topic": "lawyers", "joke_count": 3})
    # print(joke_prompt_messages)

    result = model.invoke(joke_prompt_messages)
    print(result.content)

if __name__ == "__main__":
    prompt_templates()
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnableLambda

# Load environment variables from .env
load_dotenv()

def chains():
    print("Chains")

    # Create a ChatGoogleGenAI model
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    # Define prompt templates
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a comedian who tells jokes about {topic}."),
            ("human", "Tell me {joke_count} jokes.")
        ]
    )

    # Create the combined chain using LangChain Expression Language (LCEL)
    chain = prompt_template | model | StrOutputParser() # StrOutputParser just gets the content

    # Run the chain
    result = chain.invoke({"topic": "lawyers", "joke_count": 3})
    print(result)

def chains_runnable():
    print("Chains runnable")

    # Create a ChatGoogleGenAI model
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    # Define prompt templates
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a comedian who tells jokes about {topic}."),
            ("human", "Tell me {joke_count} jokes.")
        ]
    )

    # Define additional tasks aka Runnables via lambda functions
    uppercase_output = RunnableLambda(lambda x: x.upper())
    count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")

    # Create the combined chain using LangChain Expression Language (LCEL) + add more tasks aka Runnables
    chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words

    # Run the chain
    result = chain.invoke({"topic": "lawyers", "joke_count": 3})

    # Output
    print(result)


if __name__ == "__main__":
    # chains()
    chains_runnable()
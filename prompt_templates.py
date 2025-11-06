from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def prompt_templates():
    print("Prompt templates")

    # Create a ChatGoogleGenAI model
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    # Templates with multiple placeholders
    multiple_inputs = """You are a helpful AI assistant.
    Human: Tell me a {adjective} joke about {animal}.
    Assistant:"""

    prompt_multiple = ChatPromptTemplate.from_template(multiple_inputs)
    prompt = prompt_multiple.invoke({"adjective": "funny", "animal": "panda"})

    result = model.invoke(prompt)
    print(result.content)

if __name__ == "__main__":
    prompt_templates()
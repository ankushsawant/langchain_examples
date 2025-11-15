from dotenv import load_dotenv
import os

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env
load_dotenv()

def rag_basics():
    print("Rag generative AI")

    # Define the directory containing the text file and the persistent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "sample_data", "odyssey.txt")
    persistent_directory = os.path.join(current_dir, "db", "chroma_db")

    # Check if the Chroma vector store already exists
    if not os.path.exists(persistent_directory):
        print("Persistent directory does not exist. Initializing vector store...")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The sample file '{file_path}' could not be found.")

        # Read the text content from the file
        print("\n--- Loading Content from file ---")
        loader = TextLoader(file_path)
        documents = loader.load()

        # Split the document into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        # Display information about the split docs
        print("\n--- Document Chunks Information ---")
        print(f"Number of document chunks: {len(docs)}")
        # print(f"Sample chunk:\n{docs[0].page_content}\n")

        # Create embeddings
        print("\n--- Creating embeddings ---")
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        print("\n--- Finished creating embeddings ---")

        # Create the vector store and persist it automatically
        print("\n--- Creating vector store ---")
        db = Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_directory)
        print("\n--- Finished creating vector store ---")

        # Define the user's question
        query = "Who is Odysseus' wife?"

        # Retrieve relevant documents based on the query
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.9}
        )
        relevant_docs = retriever.invoke(query)

        # Display the relevant results with metadata
        print("\n--- Relevant Documents ---")
        for i, doc in enumerate(relevant_docs, start=1):
            print(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

    else:
        print("Vector store already exists. No need to initialize.")



if __name__ == "__main__":
    rag_basics()
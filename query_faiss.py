import warnings
# CORRECTED IMPORT: We need FAISS, not Chroma
from langchain_community.vectorstores import FAISS 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import pandas as pd
import torch

FAISS_DB_PATH = "./urban_dict_faiss_db" 
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "deepseek-r1:1.5b" # changable param make sure the same model is running locally in ollama

def main():
    print("Loading FAISS vector store...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device,},
        encode_kwargs={'normalize_embeddings': False}
    )
    
    try:
        vector_store = FAISS.load_local(
            folder_path=FAISS_DB_PATH, 
            embeddings=embeddings,
            allow_dangerous_deserialization=True 
        )
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        print("Please ensure the path is correct and the index files (index.faiss, index.pkl) exist.")
        return

    print("Vector store loaded successfully.")

    # --- 2. Setup Retriever ---
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5} # Retrieve top k most relevant definitions
    )

    # --- 3. Setup the Local LLM (via Ollama) ---
    llm = ChatOllama(model=LLM_MODEL, temperature=0.2)
    
    # --- 4. Setup the RAG Chain ---
    prompt_template = """
    You are an expert on Urban Dictionary slang and informal language. Use the following Urban Dictionary definitions to answer the question.

    Context from Urban Dictionary:
    {context}

    Question: {question}

    Instructions:
    - Provide a clear, accurate answer based on the Urban Dictionary definitions provided.
    - Do not show your thinking process, reasoning steps, or any text surrounded by <think> tags.
    - If the context contains multiple definitions for a term, try to synthesize them into one coherent answer.
    - If the context doesn't contain relevant information, just say "I couldn't find a definition for that in my database."

    Answer:
    """
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    # --- 5. Interactive Query Loop ---
    print("\n--- Urban Dictionary RAG is Ready ---")
    print("Ask a question, or type 'exit' to quit.")
    while True:
        question = input("\n> ")
        if question.lower() == 'exit':
            break

        print("Thinking...")
        result = qa_chain.invoke({"query": question})
        
        print("\n--- Answer ---")
        print(result["result"])
        
        print("\n--- Sources ---")
        if result.get("source_documents"):
            for doc in result["source_documents"]:
                print(f"- Word: {doc.metadata.get('word', 'N/A')} (Score: {doc.metadata.get('score', 'N/A')})")
                print(f"  Definition: {doc.page_content[:200].strip()}...")
        print("\n" + "-"*50)


if __name__ == "__main__":

    try:
        sample_data = pd.read_csv('urban_dict_cleaned.csv')
        random_words = sample_data['word'].sample(25).unique()
        
        print("\n25 Random Words from the Dataset:")
        print(", ".join(random_words))
        print("\n" + "-"*50)
    except FileNotFoundError:
        print("Note: Could not find 'urban_dict_cleaned.csv' to show random words.")

    main()
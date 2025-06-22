import warnings
import pandas as pd
import torch
from langchain_community.vectorstores import FAISS 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

FAISS_DB_PATH = "./urban_dict_faiss_db" 
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "deepseek-r1:1.5b" # change as needed

PROMPT_EXTRACT_KEYWORD = PromptTemplate.from_template(
    """From the following user question, extract the specific single word or short phrase they want a definition for. \
Respond with ONLY the keyword(s), nothing else. \
For example, if the question is 'what is the meaning of rizz', you should respond with 'rizz'. \
It is your job to identify what the user wants some words contain multiple words, example: 'twinkie wiener sandwich', twinkie wiener sandwich\
If the question is just 'hydrophoscites', you should respond with direct definition of 'hydrophoscites'.

Question: {question}
Keyword:"""
)

PROMPT_SIMPLIFY = PromptTemplate.from_template(
    """The following is a definition for the word '{word}':
    
    '{definition}'
    
    Please provide a simple, easy-to-understand explanation of this definition for someone who might find the original complex.
    Do not add any preamble like "The simplified explanation is:". Just provide the explanation.
    """
)

PROMPT_SYNTHESIZE_MULTIPLE = PromptTemplate.from_template(
    """You are an expert on Urban Dictionary slang. The user searched for '{question}' and found multiple definitions. 
    Compare all the defintions and double check if the word is same or not. if not then
    use the following definitions to provide a coherent summary that explains the different meanings or variations of the term.

    Context from Urban Dictionary:
    {context}
    
    Synthesized Summary:
    """
)

PROMPT_THEMATIC_SUMMARY = PromptTemplate.from_template(
    """You are an expert on Urban Dictionary slang. The user searched for '{question}', but it was not found. 
    However, we found these related terms. 
    
    Based on the following definitions of similar words, can you explain what concepts or themes these related words cover? 
    This might help the user understand the context of their original search.

    Related Concepts:
    {context}

    Thematic Summary:
    """
)

### --- 2. Helper Functions for Each Output Case ---

def handle_single_match(doc, llm):
    """Case 1: Exactly one definition found."""
    word = doc.metadata.get('word', 'N/A')
    definition = doc.page_content
    
    print("\n--- Definition ---")
    print(f"Word: {word}")
    print(f"Definition: {definition}")
    
    chain = PROMPT_SIMPLIFY | llm | StrOutputParser()
    print("\n--- Simple Explanation ---")
    print("Generating simple explanation...")
    simple_explanation = chain.invoke({"word": word, "definition": definition})
    print(simple_explanation)

def handle_multiple_matches(docs, search_term, llm):
    """Case 2: Multiple definitions for the same word."""
    print(f"\n--- Found {len(docs)} Definitions for '{search_term}' ---")
    
    for i, doc in enumerate(docs):
        print(f"\nDefinition {i+1}:")
        print(f"  Word: {doc.metadata.get('word', 'N/A')}")
        print(f"  Definition: {doc.page_content}")
    
    context = "\n\n".join([f"Definition {i+1}: {doc.page_content}" for i, doc in enumerate(docs)])
    chain = PROMPT_SYNTHESIZE_MULTIPLE | llm | StrOutputParser()
    
    print("\n--- Summary of Meanings ---")
    print("Generating summary...")
    summary = chain.invoke({"question": search_term, "context": context})
    print(summary)
    
def handle_no_match(docs, search_term, llm):
    """Case 3: Word does not exist, show similar words."""
    print(f"\n--- The word '{search_term}' was not found. ---")
    print("Showing the top 5 most similar words:")

    for i, doc in enumerate(docs):
        print(f"\nSimilar Word {i+1}:")
        print(f"  Word: {doc.metadata.get('word', 'N/A')}")
        print(f"  Definition: {doc.page_content}")

    context = "\n\n".join([f"Word: {doc.metadata.get('word', 'N/A')}\nDefinition: {doc.page_content}" for doc in docs])
    chain = PROMPT_THEMATIC_SUMMARY | llm | StrOutputParser()
    
    print("\n--- Thematic Summary of Related Concepts ---")
    print("Generating summary...")
    summary = chain.invoke({"question": search_term, "context": context})
    print(summary)

def main():
    print("Loading FAISS vector store...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device},
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
        return

    print("Vector store loaded successfully.")

    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    llm = ChatOllama(model=LLM_MODEL, temperature=0.2)
    
    ### NEW: Keyword extraction chain
    keyword_extraction_chain = PROMPT_EXTRACT_KEYWORD | llm | StrOutputParser()

    print("\n--- Urban Dictionary RAG is Ready ---")
    print("Ask a question, or type 'exit' to quit.")
    while True:
        user_question = input("\n> ").strip()
        if not user_question:
            continue
        if user_question.lower() == 'exit':
            break

        print("Extracting keyword...")
        ### NEW: Step 1 - Extract the keyword from the user's question
        search_term = keyword_extraction_chain.invoke({"question": user_question}).strip()
        print(f"Searching for term: '{search_term}'...")

        ### MODIFIED: Use the extracted 'search_term' for retrieval and comparison
        retrieved_docs = retriever.get_relevant_documents(search_term)
        
        exact_matches = []
        similar_matches = []
        search_term_lower = search_term.lower()
        
        for doc in retrieved_docs:
            word_in_metadata = doc.metadata.get('word', '').lower()
            if search_term_lower == word_in_metadata or word_in_metadata in search_term_lower:
                exact_matches.append(doc)
            else:
                similar_matches.append(doc)
        
        # Case 1: Exactly one definition found for the word
        if len(exact_matches) == 1:
            handle_single_match(exact_matches[0], llm)

        # Case 2: Multiple definitions found for the word
        elif len(exact_matches) > 1:
            handle_multiple_matches(exact_matches[:5], search_term, llm)

        # Case 3: The word does not exist, show similar words
        else:
            if not similar_matches:
                print(f"I couldn't find any definitions for '{search_term}' or any similar words.")
            else:
                handle_no_match(similar_matches[:5], search_term, llm)
                
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
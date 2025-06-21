import pandas as pd
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
import os
import gc

# --- Configuration ---
CLEANED_DATA_PATH = "urban_dict_cleaned.csv" 
FAISS_DB_PATH = "./urban_dict_faiss_db"  
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 5000

def create_documents_from_batch(df_batch):
    """
    Creates a list of LangChain Document objects from a DataFrame batch,
    including the vote-weighted content and rich metadata.
    """
    documents = []
    for _, row in df_batch.iterrows():
        # Create vote-weighted content for better embedding results
        score = row.get('score', 0)
        if score > 500:
            quality_prefix = "Extremely popular and highly-rated definition"
        elif score > 250:
            quality_prefix = "Popular and well-regarded definition"
        elif score > 50:
            quality_prefix = "Common definition"
        else:
            quality_prefix = "Definition"
            
        content = (
            f"{quality_prefix} for the word '{row['word']}'.\n"
            f"Definition: {row['definition']}"
        )

        metadata = {
            "word": row['word'],
            "up_votes": int(row['up_votes']),
            "down_votes": int(row['down_votes']),
            "score": int(score)
        }
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
        
    return documents

def main():
    """
    Main function to read the cleaned CSV and build the FAISS vector store.
    """
    if os.path.exists(FAISS_DB_PATH):
        print(f"FAISS DB already exists at '{FAISS_DB_PATH}'. Skipping build process.")
        print("To rebuild, please delete the directory and run this script again.")
        return

    # --- 1. Load Data in Batches ---
    try:
        df_iterator = pd.read_csv(CLEANED_DATA_PATH, chunksize=BATCH_SIZE)
        # Get total row count for the progress bar
        total_rows = len(pd.read_csv(CLEANED_DATA_PATH))
        print(f"Found cleaned data file with {total_rows:,} rows. Starting build...")
    except FileNotFoundError:
        print(f"ERROR: The cleaned data file '{CLEANED_DATA_PATH}' was not found.")
        print("Please run the `clean_data.py` script first to generate it.")
        return

    # --- 2. Setup Embeddings Model ---
    print(f"Loading embedding model: '{EMBEDDING_MODEL_NAME}'")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}, # Change to 'cuda' if you have a GPU
        encode_kwargs={'normalize_embeddings': False} # FAISS handles normalization
    )

    # --- 3. Process Data in Batches and Build FAISS Index ---
    print("Processing data and building FAISS index (this will take a while)...")
    vector_store = None
    with tqdm(total=total_rows, desc="Embedding and Indexing", unit="row") as pbar:
        for df_chunk in df_iterator:
            # Create Document objects for the current batch
            docs_batch = create_documents_from_batch(df_chunk)
            
            if not docs_batch:
                pbar.update(len(df_chunk))
                continue

            # Create or add to the FAISS index
            if vector_store is None:
                # First batch: create the vector store
                vector_store = FAISS.from_documents(docs_batch, embeddings)
            else:
                # Subsequent batches: add to the existing store
                vector_store.add_documents(docs_batch)
            
            pbar.update(len(df_chunk))
            
            # Clean up memory to be safe
            del df_chunk, docs_batch
            gc.collect()

    # --- 4. Save the Final Index to Disk ---
    if vector_store:
        print("\nBuild complete. Saving FAISS index to disk...")
        vector_store.save_local(FAISS_DB_PATH)
        print(f"FAISS index saved successfully to '{FAISS_DB_PATH}'!")
    else:
        print("No valid documents were processed. FAISS index was not created.")

if __name__ == "__main__":
    main()
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from typing import List, Tuple
import chromadb
import PyPDF2
import os
from concurrent.futures import ThreadPoolExecutor
import threading

# Thread-safe print function
print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


def extract_text_from_pdf(pdf_path: str) -> Tuple[str, str]:
    """
    Extracts text from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Tuple of (filename, extracted text)
    """
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            safe_print(f"Extracted text from {os.path.basename(pdf_path)}")
            return os.path.basename(pdf_path), text
    except Exception as e:
        safe_print(f"Error reading {pdf_path}: {e}")
        return os.path.basename(pdf_path), ""


def chunk_text(text: str, tokenizer: AutoTokenizer, max_tokens: int = 400, overlap_tokens: int = 40) -> List[str]:
    """
    Splits text into chunks based on token count with overlap.

    Args:
        text: Input text to be chunked
        tokenizer: Hugging Face tokenizer
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Overlapping tokens between chunks

    Returns:
        List of text chunks
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    text_length = len(tokens)
    chunks = []
    start = 0

    while start < text_length:
        end = min(start + max_tokens, text_length)
        if end < text_length:
            chunk_text = tokenizer.decode(tokens[start:end], skip_special_tokens=True)
            last_sentence_end = max(
                chunk_text.rfind('.'),
                chunk_text.rfind('!'),
                chunk_text.rfind('?')
            )
            if last_sentence_end > len(chunk_text) * 0.9:
                sub_tokens = tokenizer.encode(chunk_text[:last_sentence_end + 1], add_special_tokens=False)
                end = start + len(sub_tokens)

        chunk = tokenizer.decode(tokens[start:end], skip_special_tokens=True).strip()
        if chunk:
            chunks.append(chunk)
        start += (max_tokens - overlap_tokens)

    safe_print(f"Created {len(chunks)} token-based chunks")
    return chunks


def process_pdf(pdf_path: str, tokenizer: AutoTokenizer) -> Tuple[str, List[str]]:
    """
    Extracts text from a PDF and chunks it using a tokenizer.

    Args:
        pdf_path: Path to the PDF file
        tokenizer: Hugging Face tokenizer

    Returns:
        Tuple of (filename, list of chunks)
    """
    filename, text = extract_text_from_pdf(pdf_path)
    if text:
        chunks = chunk_text(text, tokenizer)
        safe_print(f"Created {len(chunks)} chunks from {filename}")
        return filename, chunks
    return filename, []


def process_pdfs_concurrently(pdf_paths: List[str], tokenizer: AutoTokenizer, max_workers: int = 6) -> List[
    Tuple[str, List[str]]]:
    """
    Processes multiple PDFs concurrently to extract text and chunk.

    Args:
        pdf_paths: List of PDF file paths
        tokenizer: Hugging Face tokenizer
        max_workers: Number of concurrent workers

    Returns:
        List of (filename, chunks) tuples
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pdf = {executor.submit(process_pdf, pdf_path, tokenizer): pdf_path for pdf_path in pdf_paths}
        for future in future_to_pdf:
            pdf_path = future_to_pdf[future]
            try:
                filename, chunks = future.result()
                if chunks:
                    results.append((filename, chunks))
                else:
                    safe_print(f"No chunks extracted from {pdf_path}")
            except Exception as e:
                safe_print(f"Error processing {pdf_path}: {e}")
    return results


def embed_and_store_chunks(chunks: List[str], metadata: List[dict], chroma_db_path: str,
                           model_name: str = 'multi-qa-MiniLM-L6-cos-v1',
                           collection_name: str = 'pdf_chunks') -> chromadb.Collection:
    """
    Embeds text chunks and stores them in ChromaDB with metadata.

    Args:
        chunks: List of text chunks
        metadata: List of metadata dictionaries (e.g., {'source': 'filename'})
        chroma_db_path: Directory for ChromaDB persistent storage
        model_name: Name of the sentence transformer model
        collection_name: Name of the ChromaDB collection

    Returns:
        ChromaDB collection
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True).tolist()

    os.makedirs(chroma_db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_db_path)
    try:
        collection = client.get_collection(collection_name)
    except:
        collection = client.create_collection(collection_name)

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadata,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    safe_print(f"Stored {len(chunks)} chunks in ChromaDB at {chroma_db_path}")
    return collection


def pdf_to_vector_store(pdf_paths: List[str], chroma_db_path: str, tokenizer: AutoTokenizer) -> Tuple[
    List[str], List[dict], chromadb.Collection]:
    """
    Processes PDFs and stores their chunks in ChromaDB.

    Args:
        pdf_paths: List of PDF file paths
        chroma_db_path: Directory for ChromaDB persistent storage
        tokenizer: Hugging Face tokenizer

    Returns:
        Tuple of (chunks, metadata, ChromaDB collection)
    """
    pdf_results = process_pdfs_concurrently(pdf_paths, tokenizer)
    if not pdf_results:
        safe_print("No chunks extracted from any PDFs.")
        return [], [], None

    all_chunks = []
    all_metadata = []
    for filename, chunks in pdf_results:
        all_chunks.extend(chunks)
        all_metadata.extend([{"source": filename} for _ in chunks])

    if not all_chunks:
        safe_print("No valid chunks to store.")
        return [], [], None

    collection = embed_and_store_chunks(all_chunks, all_metadata, chroma_db_path)
    return all_chunks, all_metadata, collection
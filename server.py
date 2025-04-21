from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from sentence_transformers import SentenceTransformer
import chromadb
import os
import json
import asyncio
from pdf_to_vector_store import pdf_to_vector_store, safe_print
from llm_interface import LLMInterface
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# List of PDF files with absolute paths
pdf_files = [
    "D:/CSE299/PythonProject1/data/Diet.pdf",
    "D:/CSE299/PythonProject1/data/Goal.pdf",
    "D:/CSE299/PythonProject1/data/Precausions.pdf",
    "D:/CSE299/PythonProject1/data/Scientific.pdf",
    "D:/CSE299/PythonProject1/data/Templates.pdf",
    "D:/CSE299/PythonProject1/data/Workout.pdf"
]

# ChromaDB persistent storage path
CHROMA_DB_PATH = "chroma_db"

# Initialize backend components
valid_pdf_files = [pdf for pdf in pdf_files if os.path.exists(pdf)]
if not valid_pdf_files:
    raise ValueError("No valid PDF files found. Check file paths.")

# Get API keys using os.getenv
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set.")

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

nebius_api_key = hf_token
if not nebius_api_key:
    raise ValueError("NEBIUS_API_KEY environment variable not set.")

fireworks_api_key = hf_token
if not fireworks_api_key:
    raise ValueError("FIREWORKS_API_KEY environment variable not set.")

llm = LLMInterface(hf_api_token=hf_token, gemini_api_key=gemini_api_key, nebius_api_key=nebius_api_key,
                   fireworks_api_key=fireworks_api_key)
embedding_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
all_chunks, all_metadata, collection = pdf_to_vector_store(valid_pdf_files, CHROMA_DB_PATH, llm.tokenizer)


def retrieve_top_k(query: str, collection: chromadb.Collection, model: SentenceTransformer, k: int = 3) -> List[
    Tuple[str, float, dict]]:
    """
    Retrieves top-k relevant chunks based on a query using ChromaDB.
    """
    query_embedding = model.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    documents = results['documents'][0]
    distances = results['distances'][0]
    metadatas = results['metadatas'][0]
    similarities = [(doc, max(0.0, 1 - dist), meta) for doc, dist, meta in zip(documents, distances, metadatas)]
    return similarities


def is_context_relevant(top_k_results: List[Tuple[str, float, dict]], threshold: float = 0.5) -> bool:
    """
    Checks if the retrieved context is relevant based on similarity scores.
    """
    if not top_k_results:
        return False
    max_similarity = max(score for _, score, _ in top_k_results)
    return max_similarity >= threshold


async def stream_responses(query: str, target_model: str):
    """
    Streams responses from the specified model with RAG.
    """
    # Find the target model
    model = next((m for m in llm.models if m["name"] == target_model), None)
    if not model:
        yield f"data: {json.dumps({'event': 'response_chunk', 'model': 'system', 'chunk': 'Invalid model specified.'})}\n\n"
        yield f"data: {json.dumps({'event': 'done'})}\n\n"
        return

    # Handle conversational queries

    # Retrieve context and generate response
    top_k_results = retrieve_top_k(query, collection, embedding_model, k=3)
    context = "\n".join([chunk for chunk, _, _ in top_k_results])
    use_context = is_context_relevant(top_k_results, threshold=0.5)
    prompt = llm.prompt_template.format(context=context if use_context else "", question=query)

    try:
        safe_print(f"Streaming response for model: {model['name']}")
        if model["type"] == "gemini":
            stream = model["client"].generate_content(
                prompt,
                generation_config={"max_output_tokens": 2048, "temperature": 0.5},
                safety_settings={
                    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
                },
                stream=True
            )
            full_response = ""
            for chunk in stream:
                if chunk.text:
                    safe_print(f"Gemini chunk: {chunk.text}")
                    full_response += chunk.text
                    yield f"data: {json.dumps({'event': 'response_chunk', 'model': model['name'], 'chunk': full_response})}\n\n"
                    await asyncio.sleep(0.01)
        elif model["type"] == "fireworks":
            stream = model["client"].chat.completions.create(
                model=model["name"],
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant specializing in workout, diet, and gym recommendations. For questions unrelated to these topics, respond saying you dont have permission from Aurnobb to say those ."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048,
                temperature=0.5,
                stream=True
            )
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    safe_print(f"Fireworks chunk: {chunk.choices[0].delta.content}")
                    full_response += chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'event': 'response_chunk', 'model': model['name'], 'chunk': full_response})}\n\n"
                    await asyncio.sleep(0.01)
        else:  # nebius
            stream = model["client"].chat.completions.create(
                model=model["name"],
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant specializing in workout, diet, and gym recommendations. For questions unrelated to these topics, respond conversationally without directly answering the question, and steer the conversation back to fitness, diet, or gym topics."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048,
                temperature=0.5,
                stream=True
            )
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    safe_print(f"Nebius chunk: {chunk.choices[0].delta.content}")
                    full_response += chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'event': 'response_chunk', 'model': model['name'], 'chunk': full_response})}\n\n"
                    await asyncio.sleep(0.01)
    except Exception as e:
        error_msg = f"Error in {model['name']}: {str(e)}"
        safe_print(error_msg)
        yield f"data: {json.dumps({'event': 'response_chunk', 'model': model['name'], 'chunk': error_msg})}\n\n"

    yield f"data: {json.dumps({'event': 'done'})}\n\n"


@app.get("/chat")
async def chat(query: str = Query(...), model: str = Query(...)):
    """
    Streams chat responses for a given query and target model using SSE.
    """
    return StreamingResponse(stream_responses(query, model), media_type="text/event-stream")
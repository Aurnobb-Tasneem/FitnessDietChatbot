# FitnessDietChatbot

A conversational AI for workout, diet, and gym recommendations using Gemini, Mistral, and LLaMA models.

## Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables: `HF_TOKEN`, `GEMINI_API_KEY`, `NEBIUS_API_KEY`, `FIREWORKS_API_KEY`.
4. Run the FastAPI server: `uvicorn server:app --host 0.0.0.0 --port 8000`
5. Run the Streamlit app: `streamlit run app.py`

## Usage
Ask questions like "What's a good workout for beginners?" in the Streamlit UI.

## Models
- Google Gemini-1.5-Flash
- MistralAI Mixtral-8x7B-Instruct
- Meta LLaMA-3.1-70B-Instruct

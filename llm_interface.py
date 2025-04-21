from transformers import AutoTokenizer
from huggingface_hub import InferenceClient
from langchain.prompts import PromptTemplate
from typing import List, Dict
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

def safe_print(*args, **kwargs):
    from pdf_to_vector_store import safe_print as sp
    sp(*args, **kwargs)

class LLMInterface:
    def __init__(self, hf_api_token: str = None, gemini_api_key: str = None, nebius_api_key: str = None, fireworks_api_key: str = None):
        """
        Initializes the LLM interface with Gemini, Fireworks AI, and Nebius API models.

        Args:
            hf_api_token: Hugging Face API token for tokenizer
            gemini_api_key: Google Gemini API key for serverless inference
            nebius_api_key: Nebius API key for LLaMA models
            fireworks_api_key: Fireworks AI API key for Mistral models
        """
        # Hugging Face setup (for tokenizer)
        self.hf_api_token = hf_api_token or os.getenv("HF_TOKEN")
        if not self.hf_api_token:
            raise ValueError("HF_TOKEN environment variable not set. Set HF_TOKEN or provide a token.")

        # Gemini setup
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set. Set GEMINI_API_KEY or provide a key.")
        genai.configure(api_key=self.gemini_api_key)

        # Nebius setup
        self.nebius_api_key = nebius_api_key or os.getenv("NEBIUS_API_KEY")
        if not self.nebius_api_key:
            raise ValueError("NEBIUS_API_KEY environment variable not set. Set NEBIUS_API_KEY or provide a key.")

        # Fireworks AI setup
        self.fireworks_api_key = fireworks_api_key or os.getenv("FIREWORKS_API_KEY")
        if not self.fireworks_api_key:
            raise ValueError("FIREWORKS_API_KEY environment variable not set. Set FIREWORKS_API_KEY or provide a key.")

        # Tokenizer for text truncation (using LLaMA-3.1-70B-Instruct)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct", token=self.hf_api_token)
        except Exception as e:
            safe_print(f"Error loading tokenizer: {e}")
            raise

        self.max_tokens = 4096  # Suitable for LLaMA-3.1-70B-Instruct and Mixtral-8x7B

        # Models: Gemini-1.5-Flash, Mixtral-8x7B, LLaMA-3.1-70B-Instruct
        self.models = [
            {
                "name": "google/gemini-1.5-flash",
                "display_name": "Gemini",
                "client": genai.GenerativeModel("gemini-1.5-flash"),
                "type": "gemini"
            },
            {
                "name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "display_name": "Mistral",
                "client": InferenceClient(provider="nebius", api_key=self.fireworks_api_key),
                "type": "nebius"
            },
            {
                "name": "meta-llama/Llama-3.1-70B-Instruct",
                "display_name": "Llama",
                "client": InferenceClient(provider="nebius", api_key=self.nebius_api_key),
                "type": "nebius"
            }
        ]

        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="You are a helpful assistant specializing in workout, diet, and gym recommendations. Answer questions related to these topics in a conversational style, using the provided context if relevant. For questions unrelated to workout, diet, or gym, do not provide a direct answer; instead, respond conversationally, steering the conversation back to fitness, diet, or gym topics without explicitly stating the question is out of scope.\n\nContext: {context}\n\nUser: {question}\n\nAssistant:"
        )

    def truncate_to_max_tokens(self, text: str) -> str:
        """
        Truncates text to fit within the maximum token limit.
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > self.max_tokens:
            tokens = tokens[:self.max_tokens]
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return text
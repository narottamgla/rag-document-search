# OPEN_API_KEY
import os

from dotenv import load_dotenv
from langchain_classic.chat_models import init_chat_model

load_dotenv()


class Config:
    OPEN_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = "openai-gpt-4o"

    # Document processing configuration
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # Default urls
    DEFAULT_URLS = [
    "https://lilianweng.github.io/posts/2023-06-23/agent",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video",
    ]


@classmethod
def get_llm(cls):
    """Initialize and return the LLM model instance"""
    os.environ["OPENAI_API_KEY"] = cls.OPEN_API_KEY
    return init_chat_model(cls.LLM_MODEL)

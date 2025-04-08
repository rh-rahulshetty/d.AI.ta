import os
import httpx

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def get_openai_client() -> ChatOpenAI:
    '''
    Returns OpenAI compliant client
    '''
    load_dotenv()

    OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    MODEL = os.getenv('MODEL')

    http_client = httpx.Client()

    return ChatOpenAI(
        openai_api_base=OPENAI_API_BASE,
        openai_api_key=OPENAI_API_KEY,
        temperature=0,
        top_p=1,
        http_client=http_client,
        max_tokens=512,
        model=MODEL
    )

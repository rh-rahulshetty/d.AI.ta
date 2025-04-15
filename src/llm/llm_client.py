import os

from dotenv import load_dotenv
from openai import OpenAI

from src.modals.llm_data import LLM_Client_Type


def query_openai_client(model: str, client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    '''
    Returns OpenAI compliant client
    '''
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "assistant", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        top_p=1,
        max_tokens=512,
    )
    return completion.choices[0].message.content


class LLM_Client:
    '''
    Generic class to handle calls to LLM services.
    '''
    def __init__(self, llm_type: LLM_Client_Type = LLM_Client_Type.openai):
        self.type = llm_type
        if llm_type == LLM_Client_Type.openai:
            self.__init_openai__()

    def __init_openai__(self):
        load_dotenv()
        OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.model = os.getenv('MODEL')
        self.openai_client = OpenAI(
            base_url=OPENAI_API_BASE,
            api_key=OPENAI_API_KEY
        )

    def invoke(self, sys_prompt: str, user_prompt: str):
        '''Returns the result from LLM Service'''
        if self.type == LLM_Client_Type.openai:
            return query_openai_client(
                model=self.model,
                client=self.openai_client,
                system_prompt=sys_prompt,
                user_prompt=user_prompt
            )
        raise Exception("Unknown model type!")

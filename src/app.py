import random
from dotenv import load_dotenv
import matplotlib.figure
import pandas as pd
import matplotlib

import streamlit as st
from src.modals.app_data import AppResult
from src.vectordb import VectorDBSession
from src.app_workflow import generate_response

from src.utils.logger import get_module_logger
from src.utils.commons import download_file

random.seed(10)
logger = get_module_logger(__name__)

load_dotenv()

st.set_page_config(page_title="D.AI.TA Web UI", page_icon=None, layout="centered")

# Local Variables
# Store vector_db per user session.
if 'vector_db' not in st.session_state:
    st.session_state['vector_db'] = VectorDBSession()

# Keep track of processed file names.
if 'processed_files' not in st.session_state:
    st.session_state['processed_files'] = set()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


vectordb_session = st.session_state['vector_db']

# Little hack to right justify user's chat
st.markdown(
    """
<style>
    .st-emotion-cache-1c7y2kd {
        flex-direction: row-reverse;
        text-align: right;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Streamlit App
st.title("D.AI.TA: An LLM-powered AI Agent for Data Analysis, Insights and Visualizations")

st.write("This project leverages Large Language Models to simplify data analysis by enabling natural language interactions, automated insights, and visualizations—making data accessible and actionable for both technical and non-technical users.")

cols = st.columns(4)

with cols[0]:
    st.link_button("Source Code", "https://github.com/rh-rahulshetty/d.AI.ta", icon=':material/code:', use_container_width=True)

with cols[1]:
    st.link_button("Give Feedback", "https://github.com/rh-rahulshetty/d.AI.ta/issues", icon=':material/forum:', use_container_width=True)


st.warning('This is an experimental project. Please refrain from uploading any sensitive data unless you are hosting both the service and the LLM yourself.', icon="⚠️")


urls = st.text_input("Source URLs for data separated by ,")

uploaded_files = st.file_uploader(
    "Choose files",
    accept_multiple_files=True,
    type=['csv', 'log', 'zip', 'json']
)


def write_assistant_output(content: AppResult):
    if content.generation_status is False:
        st.error(body=content.message, icon='🚨')
    else:
        st.markdown(f"**Question**: *{content.user_prompt.strip()}*")

        logger.info("Response Type: %s", type(content.model_result))

        if content.message is not None and content.message != "":
            st.write(content.message)

        if isinstance(content.model_result, pd.DataFrame):
            st.dataframe(content.model_result)
        elif isinstance(content.model_result, matplotlib.figure.Figure):
            st.pyplot(content.model_result)
        else:
            st.write(content.model_result)

        if content.code != "":
            with st.expander("Code"):
                if len(vectordb_session) > 1:
                    markdown_res = "Data Sources:\n"
                    for metadata in vectordb_session.fetch_metadata_by_ids(content.file_metadata_ids):
                        markdown_res += f'- {metadata.id}: {metadata.file_name}\n'
                    st.markdown(markdown_res)
                st.code(content.code)


with st.spinner():
    # Ingest current data into vectordb session
    if len(urls) > 0:
        for url in urls.split(","):
            if url not in st.session_state['processed_files']:
                filename = download_file(url)
                with open(filename, "rb") as fobj:
                    vectordb_session.add_file(filename, fobj.read())
                    st.session_state['processed_files'].add(url)

    for file in uploaded_files:
        if file.name not in st.session_state['processed_files']:
            vectordb_session.add_file(
                file_name=file.name,
                file_bytes=file.read()
            )
            st.session_state['processed_files'].add(file.name)


# Display chat messages from history on app rerun
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == 'user':
            st.write(message["content"])
        else:
            write_assistant_output(message["content"])


# React to user input
if prompt := st.chat_input("Type your message..."):    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.write(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = None

    # Bring immediate previous message into context
    # if prompt starts with "@prev "
    if prompt.strip().startswith("@prev "):
        previous_ai_response = st.session_state.messages[-2]['content']
        response = generate_response(
            code=previous_ai_response.code,
            user_query=prompt.replace("@prev ", " ").strip(),
            vector_db=vectordb_session,
            file_metadata_ids=previous_ai_response.file_metadata_ids
        )
    else:
        response = generate_response(
            user_query=prompt,
            vector_db=vectordb_session
        )
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        write_assistant_output(response)

    # Add assistant response to chat history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

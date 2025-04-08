FROM  docker.io/library/python:3.12.9-slim

WORKDIR /app

COPY .streamlit $HOME/.streamlit
COPY requirements.txt .

RUN python3 -m pip install -r requirements.txt \
    # this line will download chroma embedding and store in cache during build time :P
    && python -c "import chromadb;client = chromadb.Client();collection = client.create_collection('all-my-documents');collection.add(documents=['This is document1'], ids=['doc1']);results = collection.query(query_texts=['This is a query document'],n_results=1)"

COPY . .

RUN python3 -m pip install -e .

EXPOSE 8501

ENTRYPOINT ["daita"]

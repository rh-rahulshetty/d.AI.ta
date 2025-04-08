# d.AI.ta

**dAIta** stands for "**D**ata assisted **AI** for **T**rue **A**nalysis" is an LLM agent solution for Data Analysis, Insights and Visualization.

> This project is currently **under active development**. Features may change, and some parts may be incomplete. Feedback and contributions are welcome!

## ✨ Features

- RAG workflow: In-memory VectorDB for data source metadata management.
- Interactive UI in Streamlit
- Data Format Supports including CSV, JSON, Log, ZIP, URL.
- Multiple AI Agents/Workflow: Code-generation, Code Refinement, Data Source Filtering, Log Parsing and Summarizer.
- **Coming Soon** Jupyter Notebook Integration, 

## Installation

### Pre-requisites

- Python 3.11+ (For local setup)
- Docker or Podman (Optional)
- LLM: [Llama 3.1](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1)

> Currently the framework has been tested with llama-3.1 model in specific, and other models might not work out-of-the-box. But it is the future roadmap to provide general support for different models.


### Local

Follow the below steps after cloning this repository.

1. Rename `.default_env` to `.env`.

2. Deploy or use OpenAPI-compatible **Llama 3.1** service. Update the `OPENAI_API_KEY`, `OPENAI_API_BASE`, `MODEL` parameters in the `.env` file.

3. Setup Virtual Environment:
    ```
    $ python3 -m venv .venv
    $ source .venv/bin/activate
    ```

4. Install the project:
    ```
    $ pip install -e .
    ```

5. Start streamlit web app in CLI:
    ```
    $ daita
    ```
    This will open up the streamlit application in http://localhost:8501

### Container (Docker/Podman)

Follow the **step 1** and **step 2** from the Local setup to create the `.env` config file.

1. Build container image
    ```
    podman build -t daita-box -f Dockerfile
    ```

2. Run container
    ```
    podman run -v $(pwd)/.env:/app/.env:z -it -p 8501:8501 localhost/daita-box
    ```
    This will open up the streamlit application in http://localhost:8501


## Acknowledgement

This project would not be possible without research and project efforts from community:

- [Meta LLAMA 3.1](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/)
- [LIDA Research Paper](https://aclanthology.org/2023.acl-demo.11/)


## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License
This project is licensed under the MIT License.


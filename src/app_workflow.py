import pandas as pd
from matplotlib.figure import Figure

from typing import Any, List

from src.code_executor.local_executor import executor
from src.vectordb import VectorDBSession

from src.modals.app_data import AppResult, CODE_EXECUTION_RETRIES
from src.modals.file_types.base import FileMetadata
from src.modals.llm_data import TasksTag

from src.utils.logger import get_module_logger

from src.llm.llm_executor import LLMTaskExecutor

from src.llm.tasks.data_for_task import DataForTask
from src.llm.tasks.code_solver import CodeSolver
from src.llm.tasks.code_refinement import CodeRefinement
from src.llm.tasks.summarizer import Summarizer

logger = get_module_logger(__name__)


def generate_response(
    user_query: str,
    vector_db: VectorDBSession,
    code: str = None,
    file_metadata_ids: List[str] = None
) -> AppResult:
    '''
    Response to return back to user query
    '''
    filtered_results = None

    if file_metadata_ids is None:
        # Fetch results from Vector DB
        vectordb_results = vector_db.query(
            query_text=user_query
        )
        logger.info("Number of metadata files retrieved: %d", len(vectordb_results))

        # Filter out VectorDB results
        filtered_results = filter_vectordb_results_by_llm(user_query, vectordb_results)
        logger.info("Number of metadata files after filtering: %d", len(filtered_results))
    else:
        filtered_results = vector_db.fetch_metadata_by_ids(ids=file_metadata_ids)

    # Exit: Relevant data not found in vectordb
    if len(filtered_results) == 0:
        return AppResult(
            user_prompt=user_query,
            message="Could not find relevant data source.",
            generation_status=False
        )

    return code_feedback_loop(
        user_query=user_query,
        code=code,
        metadatas=filtered_results,
    )


def filter_vectordb_results_by_llm(user_query: str, results: List[FileMetadata]):
    '''
    Removes metadata that are not relevant to the query.
    '''
    llm_executor = LLMTaskExecutor()

    for metadata_result in results:
        llm_executor.add_task(DataForTask(
            file_metadata=metadata_result,
            query=user_query,
            metadata=metadata_result
        ))

    llm_executor.run_tasks()

    results = []
    for model_response in llm_executor.fetch_results(search_tags=[TasksTag.data_for_task]):
        bool_value = model_response.bool_value
        if bool_value:
            results.append(model_response.metadata)

    return results


def generate_code(user_query: str, results: List[FileMetadata]):
    llm_executor = LLMTaskExecutor()

    llm_executor.add_task(CodeSolver(
        file_metadatas=results,
        query=user_query
    ))

    llm_executor.run_tasks()
    model_response = llm_executor.fetch_results()[0]

    return model_response.text


def improve_code(code: str, user_query: str, results: List[FileMetadata]):
    llm_executor = LLMTaskExecutor()

    llm_executor.add_task(CodeRefinement(
        code=code,
        file_metadatas=results,
        query=user_query
    ))

    llm_executor.run_tasks()
    model_response = llm_executor.fetch_results()[0]

    return model_response.text


def code_feedback_loop(user_query: str, metadatas: List[FileMetadata], code: str = None) -> AppResult:
    '''
    Method to run error feedback loop to generate valid code.
    It uses ReAct strategy to improve code with refinement.
    '''
    retries_left = CODE_EXECUTION_RETRIES
    code_error = None

    # 1. Try to generate code.
    if code is None:
        # code solver task
        code = generate_code(
            user_query,
            metadatas
        )
    else:
        # code refinment llm task
        code = improve_code(
            code,
            user_query,
            metadatas
        )

    while retries_left != 0:
        try:
            logger.debug("Code Generated: %s", code)

            # 2. Try to run the generated code
            code_result = executor(
                code=code,
                file_metadatas=metadatas
            )

            # Code execution had no errors

            # 3. Generate a small summary on the result
            message = generate_data_summary(
                code_result=code_result,
                user_query=user_query
            )

            return AppResult(
                message=message,
                user_prompt=user_query,
                code=code,
                generation_status=True,
                model_result=code_result,
                file_metadata_ids=[metadata.id for metadata in metadatas]
            )

        except Exception as error:
            code_error = error
            logger.error("Found issue while executing code: %s", str(error))
            # Try to generate code to fix user error.
            code = improve_code(
                code,
                f'{user_query}.\n\nCode Error: "{str(error)}"\n\nFix the issue.',
                metadatas
            )
            retries_left -= 1

    logger.error("All code generation retries completed.")
    return AppResult(
        user_prompt=user_query,
        message="Generated code did not execute successfully.\n Latest Error:" + str(code_error),
        code=code,
        generation_status=False
    )


def generate_data_summary(code_result: Any, user_query):
    '''
    Based on the result from model, try to generate a small caption on the generated data.
    '''
    if code_result is not None:
        # Convert data into string
        data = code_result
        if isinstance(code_result, pd.DataFrame):
            # Handle conversion of pandas result
            r, c = code_result.shape
            # Let's stick with data with max size 10x10
            # to pass onto summarizer.
            if r > 10 or c > 10:
                return None
            data = data.to_string(index=False)
        elif isinstance(code_result, Figure):
            # TODO: Use vision LLM to convert
            return None
        else:
            # default convert using str
            data = str(data)

        # TODO: Handle summarization for large data
        if len(data) > 1024:
            return

        # Run Summarizer
        llm_executor = LLMTaskExecutor()

        llm_executor.add_task(Summarizer(
            data=data,
            user_query=user_query
        ))

        llm_executor.run_tasks()
        model_response = llm_executor.fetch_results()[0]

        return model_response.text

    return None

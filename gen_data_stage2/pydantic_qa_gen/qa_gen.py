import asyncio
import os
import sys
import time  # Import the time module
from enum import Enum
from typing import List
from urllib.parse import urlparse

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from tqdm import tqdm  # Use tqdm.notebook for Jupyter/IPython environments

# This line loads the environment variables from the .env file
load_dotenv("../.env")

tqdm.pandas()


class QuestionContextPair(BaseModel):
    """
    Class to encapsulate a question-context-excerpt pair.

    question: The question generated from the scientific document.
    context: This should be a chunk of text from the document itself that includes the answer to the corresponding question.
    """

    question: str
    context: str


class QuestionContextGeneration(BaseModel):
    """
    Model to encapsulate the question-context generation process
    for scientific documents.
    """

    question_answer: List[QuestionContextPair] = Field(
        ...,
        description="List of question-context pairs generated from the scientific document content.",
    )


def generate_question_answer_pairs(
    df,
    model_name="qwen3:4b",
    batch_size=1,
    output_folder="qa_gen/",
    nrows=None,
    nquestion=10,
):
    if model_name.startswith("gpt"):
        ollama_model = OpenAIModel(
            model_name,
        )
    else:
        ollama_model = OpenAIModel(
            model_name=model_name,
            provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
        )

    system_prompt = f"""
        **Role & Goal:**
        You are an expert AI specializing in Information Retrieval. Your mission is to generate high-quality question-context pairs from a given scientific document. These pairs will be used to train and evaluate retrieval systems.

        **Primary Task:**
        From the user-provided text, generate exactly {nquestion} distinct question-context pairs.

        **Requirements for Questions:**
        - **Diverse:** Ensure questions cover a wide range of topics, concepts, methodologies, and results from the text.
        - **Grounded:** Each question must be answerable **only** using the information present in the source text. Do not create questions that require external knowledge.

        **Requirements for Context:**
        - **Direct Extraction:** The context **must** be a direct quote from the source text.
        - **Self-Contained:** The context must contain all the information necessary to fully answer its corresponding question. A student should need no other information.
        - **Sufficient Length:** The context should be a substantial chunk of text (a full paragraph is ideal), not just a single sentence. It should effectively narrow the focus from the full document to a specific, informative region.

        """

    agent = Agent(
        ollama_model,
        output_type=QuestionContextGeneration,
        system_prompt=system_prompt,
    )

    if nrows:
        df = df.head(nrows)

    # Ensure the output folder exists
    output_folder = f"{output_folder}{model_name}"
    os.makedirs(output_folder, exist_ok=True)

    # Load existing data if any Parquet files exist
    existing_files = [f for f in os.listdir(output_folder) if f.endswith(".parquet")]
    if existing_files:
        processed_indices = set()
        for file in existing_files:
            file_path = os.path.join(output_folder, file)
            existing_data = pd.read_parquet(file_path)
            processed_indices.update(existing_data.index)
    else:
        processed_indices = set()

    # Get the remaining rows that have not been processed
    remaining_rows = df.loc[~df.index.isin(processed_indices)]

    # If there are no remaining rows, return the existing data
    if remaining_rows.empty:
        return pd.concat(
            [pd.read_parquet(os.path.join(output_folder, f)) for f in existing_files],
        )

    for start in tqdm(
        range(0, len(remaining_rows), batch_size),
        desc="Processing batches",
    ):
        batch = remaining_rows.iloc[start : start + batch_size]

        batch[
            [
                "questions",
                "context",
                "request_tokens",
                "response_tokens",
                "total_tokens",
                "time_taken",
            ]
        ] = batch["text"].apply(
            lambda x: pd.Series(process_dataset_gen(agent, x)),
        )

        # Save each batch to a separate Parquet file
        batch_file_name = f"batch_{int(time.time() * 1000)}.parquet"
        batch_file_path = os.path.join(output_folder, batch_file_name)
        batch.to_parquet(batch_file_path)

    # After processing all batches, read all Parquet files and combine them
    final_data = pd.concat(
        [
            pd.read_parquet(os.path.join(output_folder, f))
            for f in os.listdir(output_folder)
            if f.endswith(".parquet")
        ],
    )

    return final_data


def get_remaining_url_parts(url):
    try:
        parsed_url = urlparse(url)
        # We want the path, query, and fragment.
        # We'll concatenate them, adding '?' before query if it exists,
        # and '#' before fragment if it exists.
        remaining_parts = parsed_url.path
        if parsed_url.query:
            remaining_parts += "?" + parsed_url.query
        if parsed_url.fragment:
            remaining_parts += "#" + parsed_url.fragment
        return remaining_parts
    except Exception:
        # Handle cases where the URL might be malformed
        return None


def process_dataset_gen(agent, text):
    try:
        stime = time.time()  # Start timer for processing
        result = agent.run_sync(text)
        etime = time.time()  # End timer for processing
        qa_pair = result.output.question_answer

        if result.usage():
            usage = result.usage()
            request_tokens = usage.request_tokens
            response_tokens = usage.response_tokens
            total_tokens = usage.total_tokens
        else:
            request_tokens = response_tokens = total_tokens = None

        return (
            [pair.question for pair in qa_pair],
            [pair.context for pair in qa_pair],
            request_tokens,
            response_tokens,
            total_tokens,
            etime - stime,
        )

    except Exception as e:
        # Log the error and the problematic text
        print(f"Error processing text: {text[:200]}...")  # Print first 200 chars
        print(f"Error details: {e}")
        # Return default/error values or re-raise if you want to stop
        return [], [], None, None, None, None


def main():
    data_path = "/rhome/sawale/indus_traning/sentense_transformers/gen_data_stage3/filtered_sde_data/"
    df = pd.read_parquet(
        data_path,
        columns=["id", "url1", "title", "text", "prob_included"],
    )
    # sort rows based on 'prob_included' in descending order
    df = df.sort_values(by="prob_included", ascending=False)

    # model_name = "qwen3:4b"

    model_name = "gpt-4.1-mini"

    print(f"Original Shape of data: {df.shape}")

    # Measure filter2 execution time
    filter2_start_time = time.time()
    df = generate_question_answer_pairs(
        df,
        model_name=model_name,
        nrows=5,
        output_folder="qa_test_data_gen_v4/",
    )
    filter2_end_time = time.time()
    filter2_time = filter2_end_time - filter2_start_time
    print(f"Time taken for QA generation: {filter2_time:.4f} seconds")


if __name__ == "__main__":
    main()

import asyncio
import os
import sys
import time  # Import the time module
from enum import Enum
from typing import List
from urllib.parse import urlparse

import instructor
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from tqdm import tqdm  # Use tqdm.notebook for Jupyter/IPython environments

# This line loads the environment variables from the .env file
load_dotenv("../.env")

tqdm.pandas()


class QueryContextPair(BaseModel):
    """
    This model holds a generated question (`query`) and the verbatim text snippet (`context`) from the source document
    """

    query: str = Field(
        ...,
        description=(
            "A question that is fully answerable by the 'context'. Should be a mix of factual, list/enumeration, inferential, causal, comparative or procedural questions"
        ),
    )
    context: str = Field(
        ...,
        description=(
            "A verbatim text extract from the source document. It must be a self-contained "
            "paragraph or passage that contains all information needed to answer the 'query'."
        ),
    )


class DatasetGeneration(BaseModel):
    """
    This model acts as the top-level response schema, containing a comprehensive list
    of question-context pairs and a curated list of high-quality search terms,
    all derived from the source document.
    """

    question_context: List[QueryContextPair] = Field(
        ...,
        description=(
            "A list of diverse `QueryContextPair` objects. Ensure the pairs cover a "
            "wide range of topics, methodologies, and findings from across the entire "
            "document, not just a single section."
        ),
    )
    search_terms: List[str] = Field(
        ...,
        description=("A curated list of high-relevance search terms"),
    )


def generate_dataset(
    df,
    model_name="qwen3:4b",
    batch_size=1,
    output_folder="qa_gen/",
    nrows=None,
    nquestion=10,
):
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
                "search_terms",
                "request_tokens",
                "response_tokens",
                "total_tokens",
                "time_taken",
            ]
        ] = batch["text"].apply(
            lambda x: pd.Series(process_dataset_gen(model_name, x, nquestion)),
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


def process_dataset_gen(model_name, text, nquestion=10):

    if model_name.startswith("gpt"):
        client = instructor.from_openai(OpenAI())
    else:
        client = instructor.from_openai(
            OpenAI(
                # base_url="http://localhost:11434/v1",
                base_url="http://207.157.74.65:11435/v1",
                api_key="ollama",  # required but unused
            ),
            mode=instructor.Mode.JSON,
        )

    system_prompt = f"""
        **Your Role:** You are an expert AI specializing in creating datasets for Information Retrieval.

        **Your Task:** From the provided text, generate two outputs in the required JSON format:
        1.  A list of **approximately {nquestion}** `QueryContextPair` objects.
        2.  A list of **approximately {nquestion}** `search_terms`.

        **Requirements for Queries:**
            * **Grounded:** Every question MUST be answerable using ONLY the provided `context`. Do not use external knowledge.
            * **Diverse:** Questions must cover a wide range of topics from the text

        **Requirements for Context:**
            * **Verbatim:** The `context` MUST be a direct, verbatim quote extracted from the source text.
            - **Sufficient Length:** The context should be a substantial chunk of text (a full paragraph is ideal), not just a single sentence.

        **Requirements for Search Terms:**
            * **User Intent:** Think like a researcher. What would they type into Google Scholar or PubMed to find this paper?
            * **Specificity:** Avoid vague, single-word terms.
                * **BAD:** `treatment`, `science`
                * **GOOD:** `mRNA vaccine side effects`, `protein folding accuracy`
            * **Content & Mix:** The list must capture the document's core ideas. It must include a mix of:
                1.  **Technical Phrases:** e.g., 'carbon nanotube synthesis', 'large language model fine-tuning'
                2.  **Named Entities:** e.g., 'CRISPR-Cas9', 'Hubble Space Telescope'
                3.  **Conceptual Queries:** e.g., 'how to improve battery cycle life', 'risks of AI in healthcare'

        """

    try:
        stime = time.time()  # Start timer for processing
        user_info, completion = client.chat.completions.create_with_completion(
            model=model_name,
            response_model=DatasetGeneration,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
        )

        request_tokens = completion.usage.prompt_tokens
        response_tokens = completion.usage.completion_tokens
        total_tokens = completion.usage.total_tokens

        query = [pair.query for pair in user_info.question_context]
        contexts = [pair.context for pair in user_info.question_context]
        search_terms = user_info.search_terms
        etime = time.time()  # End timer for processing

        return (
            query,
            contexts,
            search_terms,
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
        return [], [], [], None, None, None, None


def main():
    data_path = "/rhome/sawale/indus_traning/sentense_transformers/gen_data_stage3/filtered_sde_data/"
    df = pd.read_parquet(
        data_path,
        columns=["id", "url1", "title", "text", "prob_included"],
    )
    df = df.sort_values(by="prob_included", ascending=False)
    print(f"Original Shape of data: {df.shape}")

    # model_name= "hf.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF:Q4_K_M"
    # model_name = "gpt-4o-mini"
    # model_name = "qwen3:4b"
    # model_name = "gpt-4.1-mini"
    # model_name = "gemma3:4b"
    model_name = "llama3.2:3b"

    # Measure filter2 execution time
    st = time.time()
    df = generate_dataset(df, model_name=model_name, nrows=5, output_folder="data_v1/")
    et = time.time()
    time_taken = et - st
    print(f"Time taken for QA generation: {time_taken:.4f} seconds")


if __name__ == "__main__":
    main()

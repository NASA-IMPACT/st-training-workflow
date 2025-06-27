import asyncio
import os
import sys
import time  # Import the time module
from enum import Enum
from urllib.parse import urlparse

import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from tqdm.notebook import tqdm  # Use tqdm.notebook for Jupyter/IPython environments

tqdm.pandas()


class Quality(Enum):
    """
    Represents the assessed quality level of scientific document content
    for its suitability in generating high-quality question-answer pairs.
    """

    VERY_GOOD = "VERY_GOOD"
    GOOD = "GOOD"
    POOR = "POOR"


class ContentQuality(BaseModel):
    """
    Pydantic model to encapsulate the quality assessment of a scientific document's
    text content, specifically for its utility in information retrieval and
    question generation.

    This model provides a structured output for the LLM's evaluation, including
    a categorical quality rating and detailed reasoning.
    """

    quality: Quality = Field(
        ...,
        description=(
            "The overall quality rating of the scientific document content. "
            "Choose from VERY_GOOD, GOOD, or POOR, based on its clarity, "
            "information density, relevance, accuracy, and structure for "
            "extracting high quality question-answer pairs."
        ),
    )
    reasoning_traces: list[str] = Field(
        ...,
        description=(
            "A list of concise and accurate reasoning steps or bullet points "
            "that justify the assigned quality. These traces should highlight "
            "specific aspects of the content (e.g., 'Clear explanation of methodology', "
            "'Lacks sufficient factual details for 10 questions')."
        ),
    )


def filter1(df):
    df["n_words"] = df["text"].apply(lambda x: len([w for w in x.split(" ")]))
    median_word_count = df["n_words"].median()
    threshold = median_word_count // 2
    df = df[df["n_words"] > threshold]

    return df


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


def filter1_1(df):
    # it check urls
    stop_words = ["photo", "gallery", "photos", "contact", "figure", "figures"]

    df["url2"] = df["url1"].apply(get_remaining_url_parts)
    df = df[~df["url2"].str.contains("|".join(stop_words), case=False, na=False)]

    return df


def process_text_quality(agent, text):
    try:
        stime = time.time()  # Start timer for processing
        result = agent.run_sync(text)
        etime = time.time()  # End timer for processing
        quality = str(result.output.quality)
        reason = result.output.reasoning_traces

        if result.usage():
            usage = result.usage()
            request_tokens = usage.request_tokens
            response_tokens = usage.response_tokens
            total_tokens = usage.total_tokens
        else:
            request_tokens = response_tokens = total_tokens = None

        return (
            quality,
            reason,
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
        return "ERROR", [f"Failed to process: {e}"], None, None, None, None


def filter2(df, model_name="qwen3:4b", nrows=None, batch_size=4, output_folder="data/"):
    ollama_model = OpenAIModel(
        model_name=model_name,
        provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
    )

    # ollama_model = OpenAIModel(
    #     "gpt-4o-mini"
    # )

    agent = Agent(
        ollama_model,
        output_type=ContentQuality,
        system_prompt=(
            "You are an expert AI assistant specialized in critically assessing "
            "the quality of scientific research documents. Your primary goal "
            "is to evaluate how suitable a given document is for generating "
            "a high-quality dataset of question-answer pairs, where "
            "each question can be directly answered by information present "
            "within the core scientific content of the document. "
            "\n\n"
            "**Important Note on Input Text:** Be aware that the input text "
            "might originate from web scraping and could contain some residual "
            "irrelevant content like headers, footers, navigation links, or "
            "sidebars, even after initial cleaning. Your assessment should "
            "focus primarily on the *main body* of the scientific text. "
            "If irrelevant text is present, consider how severely it impacts "
            "the extractability of the core scientific information and your "
            "ability to confidently generate factual questions. If the scientific "
            "content is severely diluted or obscured by noise, this should "
            "negatively affect the quality rating.\n"
            "\n"
            "Your assessment should consider the following key aspects of the *core scientific content*:\n"
            "1.  **Information Density & Factual Richness:** Does the document "
            "contain a wealth of explicit facts, definitions, processes, "
            "results, and conclusions? Is there enough concrete information "
            "to reliably formulate at least 10 distinct, answerable questions?\n"
            "2.  **Clarity, Coherence, and Readability:** Is the language clear, "
            "unambiguous, and easy to understand? Does the text flow logically, "
            "without abrupt topic shifts or convoluted sentences? Is it free "
            "from excessive jargon or poorly explained concepts?\n"
            "3.  **Relevance and Focus:** Does the document maintain a clear "
            "focus on a specific scientific topic? Is the content cohesive "
            "and directly related to the core subject?\n"
            "4.  **Potential for Question Generation:** Beyond general quality, "
            "is the document's content diverse enough in its factual statements "
            "to inspire a minimum of 10 varied questions, each with a verifiable "
            "answer within the text?"
            "\n"
            "Based on these criteria, provide a 'quality' rating (VERY_GOOD, GOOD, POOR) "
            "and a list of 'reasoning_traces' outlining the specific reasons for your judgment. "
            "Be precise in your reasoning, citing examples or general observations "
            "about the document's content, and how any detected irrelevant text "
            "affected the overall assessment."
        ),
    )

    if nrows:
        df = df.head(nrows)
        # df = df.sample(n=nrows, random_state=42)  # Sample nrows randomly for testing

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
                f"quality",
                f"quality_reasons",
                "request_tokens",
                "response_tokens",
                "total_tokens",
                "time_taken",
            ]
        ] = batch["text"].progress_apply(
            lambda x: pd.Series(process_text_quality(agent, x)),
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


def main():
    total_start_time = time.time()  # Start total timer

    data_path = "/rhome/sawale/indus_traning/mlm-fine-tuning/mlm/data/cleaned_prod_dump_filtered_word_count_yake.parquet"
    df = pd.read_parquet(data_path, columns=["id", "url1", "title", "text"])

    model_name = "qwen3:4b"

    print(f"Original Shape of data: {df.shape}")

    # Measure filter1 execution time
    filter1_start_time = time.time()
    df = filter1(df)

    print(f"Shape of data after filter1: {df.shape}")
    filter1_end_time = time.time()
    filter1_time = filter1_end_time - filter1_start_time
    print(f"Time taken for filter1: {filter1_time:.4f} seconds")

    # url based filter
    filter1_1_start_time = time.time()
    df = filter1_1(df)

    df.to_parquet("subsample.parquet")

    print(f"Shape of data after filter1_1: {df.shape}")
    filter1_1_end_time = time.time()
    filter1_1_time = filter1_1_end_time - filter1_1_start_time
    print(f"Time taken for filter1_1: {filter1_1_time:.4f} seconds")

    # Measure filter2 execution time
    filter2_start_time = time.time()
    df = filter2(df, model_name=model_name, nrows=5, output_folder="data_v3/")
    filter2_end_time = time.time()
    filter2_time = filter2_end_time - filter2_start_time
    print(f"Time taken for filter2 (nrows=3): {filter2_time:.4f} seconds")

    df.to_csv("test2.csv")

    total_end_time = time.time()  # End total timer
    total_execution_time = total_end_time - total_start_time
    print(f"Total script execution time: {total_execution_time:.4f} seconds")


if __name__ == "__main__":
    main()

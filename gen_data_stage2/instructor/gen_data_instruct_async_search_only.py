import asyncio
import os
import time
from typing import List

import httpx
import instructor
import pandas as pd

# Load environment variables from a .env file
# Make sure your .env file has OPENAI_API_KEY="your-key-here"
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

load_dotenv(override=True)


class DatasetGeneration(BaseModel):
    """
    This model acts as the top-level response schema, containing a comprehensive list
    of high-quality search terms, all derived from the source document.
    """

    search_terms: List[str] = Field(
        ...,
        description=("A curated list of high-relevance search terms"),
    )


# --- Asynchronous Processing Function (Updated for token usage) ---
async def async_process_dataset_gen(
    client: instructor.AsyncInstructor,
    model_name: str,
    text: str,
    min_nq: int,
    max_nq: int,
    semaphore: asyncio.Semaphore,
    title=None,
):
    """
    Asynchronously generates a dataset for a single text entry.
    Uses a semaphore to limit concurrent API calls and extracts token usage.
    """
    system_prompt = f"""
        **Your Task:** From the provided text, generate a list of `search_terms` in the required JSON format. These search terms are intended to find relevant **datasets** within a scientific search engine for data repository.

        **Quantity Guidelines:**
        The ideal quantity for the list is **between {min_nq} and {max_nq}**. Your goal is to produce the highest possible quality output. Therefore, **intelligently determine the appropriate number of items within this {min_nq}-{max_nq} range based on the richness and density of the provided text.**

        **Requirements for Search Terms:**
        * **User Intent (Dataset Focused):** Think like a scientist or researcher looking for specific data. What precise terms or phrases would they use in a planetary data search engine to locate the datasets described in the text?
        * **Specificity & Precision for Data Discovery:** Avoid vague, single-word terms. Prioritize scientific, technical, and mission-specific terminology that points to datasets.
            * **BAD:** `space`, `data`
            * **GOOD:** `Mars Perseverance rover landing site selection`, `JWST exoplanet transit spectroscopy`, `Hubble Space Telescope raw images of Jupiter`
        * **Judicious Use of Keywords:** Do not append "data" or "dataset" to every term. Only include these keywords when they are essential for clarifying the user's intent to find a dataset (e.g., "solar wind turbulence datasets").

        **Content & Mix for Dataset Retrieval:**
        The list must capture the core scientific and technical concepts that would be indexed as part of a dataset. This includes a mix of:
        * **Dataset Names & Variations:** If the dataset name is mentioned in the text, include it and potentially a more natural, canonical version of it as a search term.
        * **Technical Concepts:** (e.g., `solar energetic particles`, `coronal mass ejection simulations`)
        * **Named Entities:** (e.g., `Parker Solar Probe`, `MAVEN spacecraft`)
        * **Mission and Instrument Specifics:** (e.g., `PSP/WISPR instrument observations`, `STEREO-A COR2 coronagraph Level-2 FITS`)
        * **Data Descriptors:** (e.g., `time-series plasma measurements`, `magnetogram`, `FITS file image collections`)
        * **Conceptual Queries for Datasets:** (e.g., `heliospheric magnetic field measurements`, `solar wind turbulence studies`)

        Create high-quality search terms specifically tailored for dataset information retrieval.

        """

    if title:
        new_text = f"Title: {title}\n\n{text}"
    else:
        new_text = text

    async with semaphore:
        try:
            stime = time.time()
            # The 'instructor' library attaches the raw API response to the Pydantic model
            user_info = await client.chat.completions.create(
                model=model_name,
                response_model=DatasetGeneration,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": new_text},
                ],
            )
            etime = time.time()

            # Extract data from the Pydantic model
            search_terms = user_info.search_terms

            # Safely extract token usage from the raw response
            # This works for OpenAI models; local models might not provide it.
            usage = getattr(user_info._raw_response, "usage", None)
            request_tokens = getattr(usage, "prompt_tokens", None)
            response_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)

            return pd.Series(
                [
                    search_terms,
                    request_tokens,
                    response_tokens,
                    total_tokens,
                    etime - stime,
                ],
            )

        except Exception as e:
            print(f"Error processing text: {text[:100]}... | Error: {e}")
            return pd.Series([[], None, None, None, None])


# --- Main Asynchronous Function (Updated for Universal Client) ---
async def generate_dataset_with_resume(
    df: pd.DataFrame,
    model_name: str,
    output_folder: str = "qa_gen/",
    text_column="text",
    title_column: str = None,
    nrows: int = None,
    min_nq: int = 5,
    max_nq: int = 15,
    batch_size: int = 50,
    concurrency_limit: int = 10,
):
    """
    Generates a dataset by processing a DataFrame concurrently in batches,
    saving each batch and allowing the process to be resumed.
    Works with both OpenAI and local models.
    """
    if nrows:
        df = df.head(nrows)

    output_path = f"{output_folder}{model_name.replace('/', '_')}/"  # Sanitize model name for path
    os.makedirs(output_path, exist_ok=True)

    # --- Resume Logic (Unchanged) ---
    processed_indices = set()
    try:
        existing_files = [f for f in os.listdir(output_path) if f.endswith(".parquet")]
        if existing_files:
            print(f"Found {len(existing_files)} existing batch files. Resuming...")
            for file in existing_files:
                existing_df = pd.read_parquet(os.path.join(output_path, file))
                processed_indices.update(existing_df.index)
            print(f"Loaded {len(processed_indices)} already processed entries.")
    except FileNotFoundError:
        print("No existing data found. Starting a new process.")

    remaining_df = df[~df.index.isin(processed_indices)]
    if remaining_df.empty:
        print("All entries have already been processed.")
        all_data = pd.concat(
            [pd.read_parquet(os.path.join(output_path, f)) for f in existing_files],
        )
        return all_data
    print(f"Processing {len(remaining_df)} remaining entries...")

    # --- UNIVERSAL CLIENT and Semaphore Setup ---
    print(f"Initializing client for model: {model_name}")
    if model_name.startswith("gpt-"):
        # For OpenAI models, uses API key from environment variables
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. Please set it in your .env file.",
            )
        aclient = instructor.from_openai(AsyncOpenAI())
        print("Using OpenAI client.")
    else:
        # For local models like Ollama
        aclient = instructor.from_openai(
            AsyncOpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",  # Required by library, but not used by Ollama
                http_client=httpx.AsyncClient(
                    timeout=120.0,
                ),  # Longer timeout for local models
            ),
        )
        print("Using local model client (Ollama).")

    semaphore = asyncio.Semaphore(concurrency_limit)

    # --- Process in Batches (Unchanged) ---
    for start in tqdm(
        range(0, len(remaining_df), batch_size),
        desc="Processing Batches",
    ):
        end = start + batch_size
        batch_df = remaining_df.iloc[start:end]
        tasks = [
            async_process_dataset_gen(
                aclient,
                model_name,
                row[text_column],
                min_nq,
                max_nq,
                semaphore,
                title=row[title_column] if title_column else None,
            )
            for _, row in batch_df.iterrows()
        ]
        results = await tqdm_asyncio.gather(
            *tasks,
            desc=f"Batch {start//batch_size + 1}",
        )

        results_df = pd.DataFrame(results, index=batch_df.index)
        results_df.columns = [
            "search_terms",
            "request_tokens",
            "response_tokens",
            "total_tokens",
            "time_taken",
        ]
        processed_batch = batch_df.join(results_df)

        batch_file_name = f"batch_{int(time.time() * 1000)}.parquet"
        processed_batch.to_parquet(os.path.join(output_path, batch_file_name))

    print("\nAll batches processed successfully.")
    all_files = [
        os.path.join(output_path, f)
        for f in os.listdir(output_path)
        if f.endswith(".parquet")
    ]
    final_df = pd.concat([pd.read_parquet(f) for f in all_files])

    return final_df


async def main():
    """Main execution function."""
    # data_path = "/rhome/sawale/indus_traning/sentense_transformers/gen_data_stage3/filtered_sde_data/"
    # data_path = "/rhome/sawale/indus_traning/sentense_transformers/gen_data_stage3/samplw_2k_per_division.parquet"
    # data_path = "/rhome/sawale/indus_traning/sentense_transformers/data/stage2_sde/cmr_pairs.parquet"

    data_path = "/rhome/sawale/indus_traning/sentense_transformers/data/stage2_sde/pds_pairs_structured.parquet"  # Adjust the path to your dataset
    df = pd.read_parquet(
        data_path,
        columns=["query", "context", "type", "synthesized", "source", "metadata"],
    )

    # df = pd.read_parquet(data_path, columns=["id", "url1", "title", "text", "prob_included"])
    # df = df.sort_values(by='prob_included', ascending=False)
    print(f"Original Shape of data: {df.shape}")

    # --- CHOOSE YOUR MODEL ---
    # For OpenAI (ensure OPENAI_API_KEY is in your .env file)
    # model_name = "llama3.2:3b"
    model_name = "gpt-4o-mini"

    # For a local model via Ollama
    # model_name = "qwen2:1.5b"

    st = time.time()
    processed_df = await generate_dataset_with_resume(
        df,
        model_name=model_name,
        nrows=None,
        output_folder="data_v5/",
        text_column="context",  # Change to the column containing the text to process
        title_column="query",  # Optional, if you want to include titles
        batch_size=100,
        concurrency_limit=50,  # OpenAI rate limits are often higher, you might increase this
        min_nq=3,
        max_nq=6,
    )
    et = time.time()

    print("\n--- Processing Complete ---")
    print(f"Final combined DataFrame shape: {processed_df.shape}")
    print(f"Time taken for QA generation: {et - st:.4f} seconds")
    print("\nSample of generated data:")
    print(processed_df[["time_taken", "total_tokens"]].dropna().head())


if __name__ == "__main__":
    asyncio.run(main())

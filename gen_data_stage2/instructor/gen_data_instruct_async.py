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

# --- Pydantic Models (Unchanged) ---
class QueryContextPair(BaseModel):
    """
    This model holds a generated question (`query`) and the verbatim text snippet (`context`) from the source document
    """

    query: str = Field(
        ...,
        description=("A question that is fully answerable by the 'context'."),
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


# --- Asynchronous Processing Function (Updated for token usage) ---
async def async_process_dataset_gen(
    client: instructor.AsyncInstructor,
    model_name: str,
    text: str,
    min_nq: int,
    max_nq: int,
    semaphore: asyncio.Semaphore,
):
    """
    Asynchronously generates a dataset for a single text entry.
    Uses a semaphore to limit concurrent API calls and extracts token usage.
    """
    system_prompt = f"""
    **Your Task:** From the provided text, generate two outputs in the required JSON format:
    1. A list of `QueryContextPair` objects.
    2. A list of `search_terms`.

    **Quantity Guidelines:**
    Generate a list of `QueryContextPair` objects and `search_terms` each. The ideal quantity for each list is **between 5 and 15**. Your goal is to produce the highest possible quality output. Therefore, **intelligently determine the appropriate number of items within this 5-15 range based on the richness and density of the provided text.**

    **Requirements for Queries (within QueryContextPair):**
    * **Grounded:** Every question MUST be answerable using ONLY the provided `context`. Do not use external knowledge.
    * **Domain Specificity:** Queries should reflect the types of questions a researcher or scientist would ask when using the NASA Science Discovery Engine. Focus on scientific missions, data, instruments, research outcomes and novel concepts.
    * **Generate Standalone Search Queries:** Your primary goal is to generate queries that function as precise, self-contained search questions a scientist would type. Imagine each query will be used to pull the single correct `context` from a massive database of millions of scientific documents. The query must be specific enough to succeed on its own.
        * **Mental Model:** Do not think of this as a conversation. Think of it as a scientist typing a full question into the NASA Science Discovery Engine search bar. The query must contain all the information needed to find the answer.
        * **Rule - Be Unambiguous:** To make a query self-sufficient, you must eliminate all ambiguity by replacing vague terms with the specific entities they refer to.
            * **AVOID PRONOUNS:** Replace `it`, `its`, `they`, `their`.
            * **AVOID DEMONSTRATIVES:** Replace `this`, `that`, `these`, `those`.
            * **AVOID GENERIC NOUNS:** Replace terms like `the instrument`, `the data`, `the proposal`, or `the observations` with their exact names from the text.
        * **Examples of Ambiguous vs. Search-Ready Queries:**
            * **AMBIGUOUS (like a chat message):** "Describe *its* primary mission objective."
            * **SEARCH-READY (a complete query):** "Describe the primary mission objective of the Europa Clipper."
            * **AMBIGUOUS (relies on prior context):** "What is the purpose of obtaining background observations during *the proposal*?"
            * **SEARCH-READY (contains the full context):** "What is the purpose of obtaining background observations during the JWST Proposal 2511?"
            * **AMBIGUOUS (too general):** "What were the results from *those observations*?"
            * **SEARCH-READY (specific and complete):** "What were the scientific results from the MIRI medium-resolution spectroscopy observations of the galaxy z8_GND_5296?"
    * **Diverse Types:** Questions must be a mix of various types, including Factual, List/Enumeration, Inferential, Causal, Comparative, Procedural, and Data-centric.
    * **Diverse Phrasing:** Avoid an overreliance on 'wh' questions (e.g., who, what, when, where, why, how). Ensure a variety of starting words and structures that mirror scientific inquiry.
    * **Comprehensive Coverage:** Questions should cover a wide range of relevant scientific and technical topics from the text.

    **Requirements for Context (within QueryContextPair):**
    * **Verbatim:** The `context` MUST be a direct, verbatim quote extracted from the source text.
    * **Sufficient Length:** The context should be a substantial chunk of text (ideally a full paragraph or a few related sentences), not just a single sentence. It should fully support the answer to its corresponding query.
    * **Select Self-Contained Passages:** Just as the query must be standalone, the extracted context must also be self-contained. It should provide a complete thought and be fully understandable without needing the surrounding paragraphs from the original document.
        * **Rule:** Do not extract passages that begin with connecting phrases like "Therefore," "Because of this," or "As a result," if the cause is not included in the extracted text. Avoid passages with unresolved internal references (e.g., pronouns or phrases like "this technique" where the technique itself is not defined within the selected text).
        * **BAD EXAMPLE (incomplete thought):** "Because of these high-resolution findings, the team's confidence in the atmospheric model increased significantly." (What findings?)
        * **GOOD EXAMPLE (self-contained):** "The JWST NIRSpec instrument obtained spectra of the exoplanet's atmosphere at a resolution of R~2700. This high-resolution data confirmed the presence of methane and water vapor, increasing the team's confidence in their atmospheric model."

    **Requirements for Search Terms:**
    * **User Intent:** Think like a scientist or researcher. What specific, precise terms or phrases would they input into the NASA Science Discovery Engine to find this information?
    * **Specificity & Precision:** Avoid vague, single-word terms. Prioritize scientific, technical, and mission-specific terminology.
            * **BAD:** `space`, `data`
            * **GOOD:** `Mars Perseverance rover landing site selection`, `JWST exoplanet transit spectroscopy data`
    * **Content & Mix:** The list must capture the document's core scientific and technical ideas, including a mix of technical concepts, named entities, mission specifics, data descriptors, and conceptual queries.

    Create high-quality datasets for Information Retrieval, specifically tailored for the NASA Science Discovery Engine

    """

    async with semaphore:
        try:
            stime = time.time()
            # The 'instructor' library attaches the raw API response to the Pydantic model
            user_info = await client.chat.completions.create(
                model=model_name,
                response_model=DatasetGeneration,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
            )
            etime = time.time()

            # Extract data from the Pydantic model
            query = [pair.query for pair in user_info.question_context]
            contexts = [pair.context for pair in user_info.question_context]
            search_terms = user_info.search_terms

            # Safely extract token usage from the raw response
            # This works for OpenAI models; local models might not provide it.
            usage = getattr(user_info._raw_response, "usage", None)
            request_tokens = getattr(usage, "prompt_tokens", None)
            response_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)

            return pd.Series(
                [
                    query,
                    contexts,
                    search_terms,
                    request_tokens,
                    response_tokens,
                    total_tokens,
                    etime - stime,
                ],
            )

        except Exception as e:
            print(f"Error processing text: {text[:100]}... | Error: {e}")
            return pd.Series([[], [], [], None, None, None, None])


# --- Main Asynchronous Function (Updated for Universal Client) ---
async def generate_dataset_with_resume(
    df: pd.DataFrame,
    model_name: str,
    output_folder: str = "qa_gen/",
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
                row["text"],
                min_nq,
                max_nq,
                semaphore,
            )
            for _, row in batch_df.iterrows()
        ]
        results = await tqdm_asyncio.gather(
            *tasks,
            desc=f"Batch {start//batch_size + 1}",
        )

        results_df = pd.DataFrame(results, index=batch_df.index)
        results_df.columns = [
            "questions",
            "context",
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
    data_path = "/rhome/sawale/indus_traning/sentense_transformers/data/stage2_sde/cmr_pairs.parquet"
    df = pd.read_parquet(
        data_path,
        columns=["id", "url1", "title", "text", "prob_included"],
    )
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
        output_folder="data_v3/",
        batch_size=100,
        concurrency_limit=50,  # OpenAI rate limits are often higher, you might increase this
        min_nq=5,
        max_nq=15,
    )
    et = time.time()

    print("\n--- Processing Complete ---")
    print(f"Final combined DataFrame shape: {processed_df.shape}")
    print(f"Time taken for QA generation: {et - st:.4f} seconds")
    print("\nSample of generated data:")
    print(processed_df[["questions", "time_taken", "total_tokens"]].dropna().head())


if __name__ == "__main__":
    asyncio.run(main())

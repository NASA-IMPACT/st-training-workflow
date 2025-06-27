import multiprocessing as mp
import os
import sys
import time
from enum import Enum

import pandas as pd
from pydantic import BaseModel, Field

# Assuming pydantic_ai and its components are correctly installed
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from tqdm import tqdm  # Use standard tqdm for multiprocessing

# tqdm.pandas() # Not used directly with multiprocessing Pool's map


class Quality(Enum):
    VERY_GOOD = "VERY_GOOD"
    GOOD = "GOOD"
    POOR = "POOR"


class ContentQuality(BaseModel):
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


# Define the system prompt globally or pass it carefully
SYSTEM_PROMPT_FOR_AGENT = (
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
)

# This global variable will hold the agent specific to each worker process
agent_process_local = None


def init_worker(model_name_for_worker, base_url_for_worker, system_prompt_for_worker):
    """Initializes an Agent for each worker process."""
    global agent_process_local
    ollama_model = OpenAIModel(
        model_name=model_name_for_worker,
        provider=OpenAIProvider(base_url=base_url_for_worker),
    )
    agent_process_local = Agent(
        ollama_model,
        output_type=ContentQuality,
        system_prompt=system_prompt_for_worker,
    )


def process_text_quality_mp(item_tuple):
    """
    Processes a single text item using the process-local agent.
    item_tuple is expected to be (index, text_content).
    """
    item_index, text_content = item_tuple
    global agent_process_local
    if agent_process_local is None:
        # This should not happen if initializer is called correctly
        print(
            f"Error: Agent not initialized in worker process for item index {item_index}.",
        )
        return item_index, "ERROR_AGENT_INIT", ["Agent not initialized"]
    try:
        result = agent_process_local.run_sync(
            text_content,
        ).output  # .run_sync and .output are from your original code
        quality = str(result.quality.value)  # Get the string value of the Enum
        reason = result.reasoning_traces
        return item_index, quality, reason
    except Exception as e:
        print(f"Error processing text (index {item_index}): {text_content[:200]}...")
        print(f"Error details: {e}")
        return item_index, "ERROR_PROCESSING", [f"Failed to process: {e}"]


def filter1(df):
    # Ensure 'text' column is string type to prevent errors with .split()
    df["text"] = df["text"].astype(str)
    df["n_words"] = df["text"].apply(lambda x: len(x.split()))
    median_word_count = df["n_words"].median()
    # Ensure threshold is at least 1, or handle cases where median_word_count is small
    threshold = max(1, int(median_word_count // 2))
    df_filtered = df[
        df["n_words"] > threshold
    ].copy()  # Use .copy() to avoid SettingWithCopyWarning
    return df_filtered


def filter2_mp(
    df,
    model_name="qwen3:4b",
    nrows=None,
    save_batch_size=100,
    num_processes=4,
    output_folder="data/",
    ollama_base_url="http://localhost:11434/v1",
):
    """
    Processes the DataFrame using multiprocessing.
    - save_batch_size: How many items to process before saving an intermediate file.
    - num_processes: Number of worker processes (ideally <= number of GPUs).
    """
    if nrows:
        df_to_process = df.head(nrows).copy()
    else:
        df_to_process = df.copy()

    # Ensure the index is named for easier merging later if it's reset
    if df_to_process.index.name is None:
        df_to_process.index.name = "original_index"

    output_folder_path = os.path.join(
        output_folder,
        model_name.replace(":", "_"),
    )  # Sanitize model name for folder
    os.makedirs(output_folder_path, exist_ok=True)

    processed_indices = set()
    existing_files = [
        f for f in os.listdir(output_folder_path) if f.endswith(".parquet")
    ]
    if existing_files:
        for file in existing_files:
            file_path = os.path.join(output_folder_path, file)
            try:
                existing_data = pd.read_parquet(file_path)
                processed_indices.update(existing_data.index)
            except Exception as e:
                print(
                    f"Warning: Could not read or process existing file {file_path}: {e}",
                )

    remaining_rows = df_to_process.loc[~df_to_process.index.isin(processed_indices)]

    if remaining_rows.empty:
        print("No new rows to process. Loading existing data.")
        if existing_files:
            final_df_list = []
            for f_path in [os.path.join(output_folder_path, f) for f in existing_files]:
                if os.path.exists(f_path) and os.path.getsize(f_path) > 0:
                    try:
                        final_df_list.append(pd.read_parquet(f_path))
                    except Exception as e:
                        print(f"Warning: Could not read {f_path} during pre-check: {e}")
            if not final_df_list:
                return pd.DataFrame(
                    columns=df_to_process.columns.tolist()
                    + ["quality", "quality_reasons"],
                )
            return pd.concat(final_df_list).sort_index()
        else:
            return pd.DataFrame(
                columns=df_to_process.columns.tolist() + ["quality", "quality_reasons"],
            )

    # Prepare tasks for multiprocessing: list of (index, text) tuples
    tasks = [(index, row["text"]) for index, row in remaining_rows.iterrows()]
    num_tasks = len(tasks)

    print(
        f"Starting processing of {num_tasks} remaining rows with {num_processes} processes.",
    )

    # Use mp.Pool for parallel processing
    # The SYSTEM_PROMPT_FOR_AGENT will be inherited by child processes if defined globally before Pool creation.
    # For more explicit control, pass it via initargs.
    with mp.Pool(
        processes=num_processes,
        initializer=init_worker,
        initargs=(model_name, ollama_base_url, SYSTEM_PROMPT_FOR_AGENT),
    ) as pool:
        for i in tqdm(
            range(0, num_tasks, save_batch_size),
            desc="Processing and Saving Batches",
        ):
            current_task_chunk_data = tasks[i : i + save_batch_size]
            if not current_task_chunk_data:
                continue

            # `pool.map` applies `process_text_quality_mp` to each item in `current_task_chunk_data`
            # It blocks until all tasks in this chunk are done.
            try:
                chunk_results = pool.map(
                    process_text_quality_mp,
                    current_task_chunk_data,
                )
            except Exception as e:
                print(f"Error in pool.map for a chunk: {e}")
                # Decide how to handle: skip chunk, retry, or stop
                continue  # Skip this chunk

            # Prepare data for DataFrame
            results_for_df = []
            valid_indices_in_chunk = []
            for result_tuple in chunk_results:
                if (
                    result_tuple
                ):  # Ensure result is not None (e.g. if an error returned None)
                    res_idx, quality, reason = result_tuple
                    results_for_df.append(
                        {"quality": quality, "quality_reasons": reason},
                    )
                    valid_indices_in_chunk.append(res_idx)
                else:
                    print("Warning: A process returned a None result.")

            if not results_for_df:
                print(f"No valid results obtained for batch starting at index {i}.")
                continue

            # Create a DataFrame from the results for this chunk
            batch_results_df = pd.DataFrame(
                results_for_df,
                index=pd.Index(valid_indices_in_chunk, name=remaining_rows.index.name),
            )

            # Get the original data for this chunk using valid_indices_in_chunk
            # and merge/join the new quality columns.
            # Ensure remaining_rows.loc[valid_indices_in_chunk] doesn't reorder if valid_indices_in_chunk is not sorted.
            # It's safer to join.
            original_chunk_df = remaining_rows.loc[valid_indices_in_chunk].copy()

            # Assign new columns to the original data chunk
            # This ensures all original columns are preserved.
            # Using .join is safer if indices align perfectly.
            # If batch_results_df has the same index as original_chunk_df (it should by design):
            batch_to_save = original_chunk_df.join(batch_results_df)

            # Save this processed batch
            timestamp = int(time.time() * 1000)
            batch_file_name = f"batch_mp_{timestamp}_{i // save_batch_size}.parquet"
            batch_file_path = os.path.join(output_folder_path, batch_file_name)
            try:
                batch_to_save.to_parquet(batch_file_path)
            except Exception as e:
                print(f"Error saving batch file {batch_file_path}: {e}")

    print(f"Finished processing all items.")

    # Consolidate all saved Parquet files
    all_saved_files = [
        os.path.join(output_folder_path, f)
        for f in os.listdir(output_folder_path)
        if f.endswith(".parquet")
    ]

    if not all_saved_files:
        print("No parquet files found after processing to combine.")
        # Return an empty DataFrame with expected columns if nothing was processed or saved
        return pd.DataFrame(
            columns=df_to_process.columns.tolist() + ["quality", "quality_reasons"],
        )

    final_df_list = []
    for f_path in all_saved_files:
        if (
            os.path.exists(f_path) and os.path.getsize(f_path) > 0
        ):  # Check if file is not empty
            try:
                final_df_list.append(pd.read_parquet(f_path))
            except Exception as e:
                print(
                    f"Warning: Could not read or process file {f_path} during final concat: {e}",
                )

    if not final_df_list:
        print("No data loaded from parquet files for final concatenation.")
        return pd.DataFrame(
            columns=df_to_process.columns.tolist() + ["quality", "quality_reasons"],
        )

    final_data = pd.concat(final_df_list)

    # Deduplicate based on index (original DataFrame index name)
    # This handles cases where batches might have overlapped or if resuming after partial processing of a batch.
    if not final_data.index.is_unique:
        final_data = final_data[~final_data.index.duplicated(keep="last")]

    return final_data.sort_index()


def main_mp():
    total_start_time = time.time()

    # --- Configuration ---
    # Make sure this path is correct or use a smaller test file
    data_path = "/rhome/sawale/indus_traning/mlm-fine-tuning/mlm/data/cleaned_prod_dump_filtered_word_count_yake.parquet"
    output_csv_path = "test_multiprocess_output.csv"
    output_folder_intermediate = "data_mp_output/"  # For intermediate parquet files

    model_name = "qwen3:4b"  # Your Ollama model
    ollama_base_url = "http://localhost:11434/v1"  # Ollama API endpoint

    # Set to the number of GPUs you have and want to use
    # Ensure Ollama can handle this many concurrent requests.
    num_processes = 4  # e.g., 4 for 4 GPUs

    # How many rows to process in total for this run (None for all)
    # For testing, set to a small number like 20 or 50.
    # nrows_to_process = None
    nrows_to_process = 50  # For quick testing

    # How many processed items to group together before saving to a parquet file.
    # This helps with memory and allows resuming if the script is interrupted.
    save_batch_size = 16  # Adjust based on item processing time and memory
    # ---------------------

    try:
        df = pd.read_parquet(data_path, columns=["id", "url1", "title", "text"])
        # Critical: Ensure 'id' is set as index if it's unique and intended for tracking.
        # The resume logic relies on a consistent and unique index.
        if "id" in df.columns:
            df = df.set_index("id", drop=False)  # Keep 'id' also as a column if needed
            if not df.index.is_unique:
                print(
                    f"Warning: Index 'id' from column is not unique. Cardinality: {df.index.nunique()} vs {len(df)}. This may affect resume logic.",
                )
                # Option: df = df.reset_index(drop=True) # if 'id' is not suitable, fall back to default range index.
        else:
            print(
                "Warning: 'id' column not found. Using existing index or default range index.",
            )
            if df.index.name is None:
                df.index.name = "original_index"

    except FileNotFoundError:
        print(
            f"Warning: Parquet file not found at {data_path}. Using a dummy DataFrame for testing.",
        )
        num_dummy_rows = max(
            nrows_to_process or 50,
            num_processes * 5,
        )  # Ensure enough data for test
        data = {
            "id": range(num_dummy_rows),
            "url1": [f"http://example.com/{i}" for i in range(num_dummy_rows)],
            "title": [f"Title {i}" for i in range(num_dummy_rows)],
            "text": [
                f"This is sample scientific text number {i}. It has enough content for testing and quality assessment. "
                * 5
                for i in range(num_dummy_rows)
            ],
        }
        df = pd.DataFrame(data).set_index("id")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if df.empty:
        print("Input DataFrame is empty. Exiting.")
        return

    print(f"Original Shape of data: {df.shape}")

    filter1_start_time = time.time()
    df_after_f1 = filter1(df)  # filter1 returns a new DataFrame
    print(f"Shape of data after filter1: {df_after_f1.shape}")
    filter1_end_time = time.time()
    print(
        f"Time taken for filter1: {filter1_end_time - filter1_start_time:.4f} seconds",
    )

    if df_after_f1.empty:
        print("DataFrame is empty after filter1. Exiting.")
        return

    filter2_start_time = time.time()
    df_processed = filter2_mp(
        df_after_f1,
        model_name=model_name,
        nrows=nrows_to_process,
        save_batch_size=save_batch_size,
        num_processes=num_processes,
        output_folder=output_folder_intermediate,
        ollama_base_url=ollama_base_url,
    )
    filter2_end_time = time.time()

    if df_processed is not None and not df_processed.empty:
        print(f"Shape of data after filter2_mp: {df_processed.shape}")
        print(
            f"Time taken for filter2_mp: {filter2_end_time - filter2_start_time:.4f} seconds",
        )
        try:
            df_processed.to_csv(
                output_csv_path,
                index=(df_processed.index.name is not None),
            )
            print(f"Output saved to {output_csv_path}")
        except Exception as e:
            print(f"Error saving final CSV: {e}")
    else:
        print("Processed DataFrame is empty or None after filter2_mp.")

    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time
    print(f"Total script execution time: {total_execution_time:.4f} seconds")


if __name__ == "__main__":
    # This is important for multiprocessing, especially on Windows.
    # It prevents child processes from re-executing the main module's code.
    mp.freeze_support()
    main_mp()

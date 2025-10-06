import asyncio
import os

import pandas as pd
import tiktoken
from datasets import load_dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI  # Import the asynchronous client
from tqdm.auto import tqdm

load_dotenv()


# %%
def get_data(
    dataset_path: str,
    corpus_split: str = "train",
    queries_split: str = "train",
):
    """Loads corpus, queries, and relevant documents from the dataset."""

    try:
        corpus_dataset = load_dataset(
            dataset_path,
            data_files="corpus.jsonl",
            split=corpus_split,
            token=os.environ["HUGGINGFACE_TOKEN"],
        )
        queries_dataset = load_dataset(
            dataset_path,
            data_files="queries.jsonl",
            split=queries_split,
            token=os.environ["HUGGINGFACE_TOKEN"],
        )
    except Exception as e:
        corpus_dataset = load_dataset(
            dataset_path,
            name="corpus",
            split=corpus_split,
            token=os.environ["HUGGINGFACE_TOKEN"],
        )
        queries_dataset = load_dataset(
            dataset_path,
            name="queries",
            split=queries_split,
            token=os.environ["HUGGINGFACE_TOKEN"],
        )

    corpus = {row["_id"]: row["text"] for row in corpus_dataset}
    queries = {row["_id"]: row["text"] for row in queries_dataset}

    return corpus, queries


# %%
# --- Configuration ---
# MODEL_NAME = "text-embedding-3-small"
# MODEL_NAME = "text-embedding-3-large"


def truncate_text(text: str, model_name, max_tokens: int = 8192) -> str:
    """Truncates a text string to a maximum number of tokens."""
    tokenizer = tiktoken.encoding_for_model(model_name)
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        return tokenizer.decode(truncated_tokens)
    return text


# --- NEW: Asynchronous and Concurrent Embedding Generation ---


async def _create_embedding_df_async(
    client: AsyncOpenAI,
    data: dict,
    model: str,
    batch_size: int,
    semaphore: asyncio.Semaphore,
    desc: str,
) -> pd.DataFrame:
    """
    Helper function to generate embeddings concurrently using asyncio.Semaphore.
    """
    ids = list(data.keys())
    texts = list(data.values())

    # This dictionary will store results, keyed by batch index to maintain order
    results_dict = {}

    async def get_embeddings_for_batch(batch_index: int, batch_texts: list[str]):
        """Worker coroutine to process one batch of texts."""
        async with semaphore:  # Wait for the semaphore to allow a new request
            cleaned_batch = [
                text if text and isinstance(text, str) and text.strip() else " "
                for text in batch_texts
            ]

            if not cleaned_batch:
                results_dict[batch_index] = []  # Store empty result for empty batch
                return

            truncated_batch = [truncate_text(text, model) for text in cleaned_batch]

            response = await client.embeddings.create(
                input=truncated_batch,
                model=model,
            )

            # Store results using the batch index
            embeddings = [res.embedding for res in response.data]
            tokens_used = response.usage.total_tokens
            results_dict[batch_index] = (embeddings, tokens_used)

    # Create a list of tasks for all batches
    tasks = []
    for i, j in enumerate(range(0, len(texts), batch_size)):
        batch = texts[j : j + batch_size]
        task = asyncio.create_task(get_embeddings_for_batch(i, batch))
        tasks.append(task)

    # Run tasks concurrently with a progress bar
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=desc):
        await future  # Wait for each task to complete

    # Process results in the correct order
    all_embeddings = []
    total_tokens_used = 0
    sorted_results = [results_dict[i] for i in sorted(results_dict.keys())]

    for embeddings_batch, tokens_in_batch in sorted_results:
        all_embeddings.extend(embeddings_batch)
        total_tokens_used += tokens_in_batch

    return pd.DataFrame(
        {
            "id": ids,
            "embeddings": all_embeddings,
            "total_tokens": [total_tokens_used] * len(ids),
        },
    )


async def gen_openai_emb_async(
    corpus: dict,
    queries: dict,
    model: str,
    batch_size: int = 50,
    concurrency_limit: int = 5,  # Max concurrent requests
):
    """
    Generates embeddings for a corpus and queries concurrently.
    """
    print("Initializing async client and semaphore...")
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    semaphore = asyncio.Semaphore(concurrency_limit)

    print("Creating concurrent tasks for corpus and query embeddings...")

    # Create two main tasks: one for the corpus, one for the queries
    corpus_task = _create_embedding_df_async(
        client,
        corpus,
        model,
        batch_size,
        semaphore,
        "Generating Corpus Embeddings",
    )
    query_task = _create_embedding_df_async(
        client,
        queries,
        model,
        batch_size,
        semaphore,
        "Generating Query Embeddings",
    )

    # Run both tasks in parallel and wait for them to complete
    corpus_embeddings_df, query_embeddings_df = await asyncio.gather(
        corpus_task,
        query_task,
    )

    return corpus_embeddings_df, query_embeddings_df


# the main function to run the embedding generation
# %%
async def generate_openai_embeddings(
    dataset_path: str,
    output_dir: str,
    model: str,
    corpus_split: str = "train",
    queries_split: str = "train",
):
    """Main function to run the data loading and embedding generation."""
    print(f"Generating OpenAI embeddings... using {model} for {dataset_path} dataset")
    # dataset_path is either a hf or local path
    corpus, queries = get_data(
        dataset_path,
        corpus_split,
        queries_split,
    )

    corpus_df, queries_df = await gen_openai_emb_async(corpus, queries, model)

    os.makedirs(output_dir, exist_ok=True)
    corpus_df.to_parquet(os.path.join(output_dir, "corpus_embeddings.parquet"))
    queries_df.to_parquet(os.path.join(output_dir, "queries_embeddings.parquet"))

    return corpus_df, queries_df

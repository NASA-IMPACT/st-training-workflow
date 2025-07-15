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
    relevant_docs_split: str = "test",
):
    """Loads corpus, queries, and relevant documents from the dataset."""

    try:
        corpus_dataset = load_dataset(
            dataset_path,
            data_files="corpus.jsonl",
            split=corpus_split,
        )
        queries_dataset = load_dataset(
            dataset_path,
            data_files="queries.jsonl",
            split=queries_split,
        )
        relevant_docs_dataset = load_dataset(dataset_path, split=relevant_docs_split)
    except Exception as e:
        corpus_dataset = load_dataset(dataset_path, name="corpus", split=corpus_split)
        queries_dataset = load_dataset(
            dataset_path,
            name="queries",
            split=queries_split,
        )
        relevant_docs_dataset = load_dataset(
            dataset_path,
            name="qrels",
            split=relevant_docs_split,
        )

    corpus = {row["_id"]: row["text"] for row in corpus_dataset}
    queries = {row["_id"]: row["text"] for row in queries_dataset}

    relevant_docs_data = (
        relevant_docs_dataset.to_pandas()
        .groupby("query-id")["corpus-id"]
        .apply(set)
        .to_dict()
    )
    relevant_docs_data = {
        str(k): {str(item) for item in v} for k, v in relevant_docs_data.items()
    }

    return corpus, queries, relevant_docs_data


# %%
# --- Configuration ---
MODEL_NAME = "text-embedding-3-small"
# MODEL_NAME = "text-embedding-3-large"
MAX_TOKENS = 8192

# --- Initialize Tokenizer ---
tokenizer = tiktoken.encoding_for_model(MODEL_NAME)


def truncate_text(text: str, max_tokens: int = MAX_TOKENS) -> str:
    """Truncates a text string to a maximum number of tokens."""
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

            truncated_batch = [truncate_text(text) for text in cleaned_batch]

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
    model: str = MODEL_NAME,
    batch_size: int = 100,
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


# %%
async def main(
    dataset_path: str,
    output_dir: str,
    model: str = MODEL_NAME,
    corpus_split: str = "train",
    queries_split: str = "train",
    relevant_docs_split: str = "test",
):
    """Main function to run the data loading and embedding generation."""
    # dataset_path = "nasa-impact/nasa-sde-IR-benchmark-sample-v1"
    corpus, queries, relevant_docs_data = get_data(
        dataset_path,
        corpus_split,
        queries_split,
        relevant_docs_split,
    )

    # Sample a small subset for testing
    # corpus_sample = {k: corpus[k] for k in list(corpus.keys())[:1000]}
    # queries_sample = {k: queries[k] for k in list(queries.keys())[:1000]}

    corpus_df, queries_df = await gen_openai_emb_async(corpus, queries)

    output_dir = os.path.join(output_dir, MODEL_NAME, dataset_path.split("/")[-1])
    os.makedirs(output_dir, exist_ok=True)
    corpus_df.to_parquet(os.path.join(output_dir, "corpus_embeddings.parquet"))
    queries_df.to_parquet(os.path.join(output_dir, "queries_embeddings.parquet"))

    return corpus_df, queries_df


if __name__ == "__main__":
    # Load the dataset
    output_dir = (
        "/rhome/sawale/indus_traning/sentense_transformers/eval/openai_emb_cache/"
    )
    # dataset_path = "nasa-impact/nasa-sde-IR-benchmark-sample-v1"
    # corpus_df, queries_df = asyncio.run(main(dataset_path, output_dir))

    # dataset_path = "nasa-impact/nasa-sde-IR-benchmark-sample-v2"
    # corpus_df, queries_df = asyncio.run(main(dataset_path, output_dir))

    # dataset_path = "nasa-impact/nasa-smd-IR-benchmark"
    # corpus_df, queries_df = asyncio.run(main(dataset_path, output_dir))

    # for nanobert
    output_dir = "/rhome/sawale/indus_traning/sentense_transformers/eval/openai_emb_cache/nanobeir"
    dataset_names = [
        "zeta-alpha-ai/NanoClimateFEVER",
        "zeta-alpha-ai/NanoDBPedia",
        "zeta-alpha-ai/NanoFEVER",
        "zeta-alpha-ai/NanoFiQA2018",  # issue
        "zeta-alpha-ai/NanoHotpotQA",
        "zeta-alpha-ai/NanoMSMARCO",
        "zeta-alpha-ai/NanoNFCorpus",
        "zeta-alpha-ai/NanoNQ",
        "zeta-alpha-ai/NanoQuoraRetrieval",
        "zeta-alpha-ai/NanoSCIDOCS",  # issue
        "zeta-alpha-ai/NanoArguAna",
        "zeta-alpha-ai/NanoSciFact",
        "zeta-alpha-ai/NanoTouche2020",
    ]

    for dataset_path in dataset_names:
        print(f"Processing dataset: {dataset_path}")
        corpus_df, queries_df = asyncio.run(
            main(
                dataset_path,
                output_dir,
                corpus_split="train",
                queries_split="train",
                relevant_docs_split="train",
            ),
        )
        print(f"Processed {dataset_path} successfully.")

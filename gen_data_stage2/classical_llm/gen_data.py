import os
import re
import time
from typing import Dict

import pandas as pd
import requests
from tqdm import tqdm

data_path = "/rhome/sawale/indus_traning/sentense_transformers/gen_data_stage3/filtered_sde_data/"
data = pd.read_parquet(
    data_path,
    columns=["id", "url1", "title", "text", "prob_included"],
)
# sort rows based on 'prob_included' in descending order
data = data.sort_values(by="prob_included", ascending=False)


def gen_qa_pair(
    text,
    url="http://localhost:11434/api/generate",
    model_name="gemma3:12b",
    max_test_len_thresh=5800,
    nquestion=10,
):
    dummy_resp = [[], [], None, None, None, None]
    qa_pair = [[], [], None, None, None, None]
    if len(text) > max_test_len_thresh:
        print(f"Text longer than max character threshold of: {max_test_len_thresh}")
        return dummy_resp

    system_message = f"""
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

        Please structure your responses in the following format:
        Question 1: <...>
        Context 1: <...>

        Question 2: <...>
        Context 2: <...>

        ...
        Question n: <...>
        Context n: <...>

        **Output Format:**
        Each question and its corresponding context should be clearly numbered and separated by a newline. Ensure there are no extra spaces or lines in the output.
    """

    payload = {
        "model": model_name,
        "system": system_message,
        "prompt": f"Text to concider:\n{text}\n\n",
        "stream": False,
        "max_tokens": 512,
        "temperature": 0.6,
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return dummy_resp  # Return a list with placeholders if request fails

    if response.status_code == 200:
        try:
            # Parse and clean the response
            resp = response.json().get("response", "").strip()

            # Corrected and improved regex
            questions_and_answers = re.findall(
                r"Question \d+: (.*?)\nContext \d+: (.*?)(?=\n\nQuestion|\Z)",
                resp,
                re.S,
            )
            for pair_index, (q, a) in enumerate(questions_and_answers):
                q = q.strip()
                a = a.strip()
                if not q or not a:
                    print(f"Skipping empty question or context at index {pair_index}")
                    continue
                qa_pair[0].append(q)
                qa_pair[1].append(a)
            qa_pair[2] = response.json().get("prompt_eval_count", None)  # requet tokens
            qa_pair[3] = response.json().get("eval_count", None)  # response tokens
            qa_pair[4] = qa_pair[2] + qa_pair[3]  # total tokens
            qa_pair[5] = response.json().get("total_duration", None)  # time taken
            # convert from ns to seconds
            if qa_pair[5] is not None:
                qa_pair[5] /= 1e9

        except (ValueError, KeyError, IndexError) as e:
            print(f"Error processing response: {e}")
            return dummy_resp  # Return placeholders if error in processing the response
    else:
        print(f"Error: Received status code {response.status_code}")
        return dummy_resp  # Return placeholders in case of non-200 response

    return qa_pair


def generate_question_answer_pairs(
    data,
    model_name="gemma3:12b",
    output_folder="../data/",
    batch_size=50,
    nrows=None,
):

    # Ensure the output folder exists
    output_folder = f"{output_folder}{model_name}"
    os.makedirs(output_folder, exist_ok=True)

    # Sample the data
    if nrows:
        sampled_data = data.head(nrows)
    else:
        sampled_data = data

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
    remaining_rows = sampled_data.loc[~sampled_data.index.isin(processed_indices)]

    # If there are no remaining rows, return the existing data
    if remaining_rows.empty:
        return pd.concat(
            [pd.read_parquet(os.path.join(output_folder, f)) for f in existing_files],
        )

    # Process and save in batches
    new_data_list = []
    for start in tqdm(
        range(0, len(remaining_rows), batch_size),
        desc="Processing batches",
    ):
        batch = remaining_rows.iloc[start : start + batch_size]

        # Generate classification metrics for the batch
        metrics_list = [
            gen_qa_pair(text, model_name=model_name) for text in batch["text"]
        ]

        # Convert to DataFrame with appropriate column names
        metrics_df = pd.DataFrame(
            metrics_list,
            columns=[
                "questions",
                "context",
                "request_tokens",
                "response_tokens",
                "total_tokens",
                "time_taken",
            ],
        )
        metrics_df.index = batch.index

        # Append metrics to batch
        batch = pd.concat([batch, metrics_df], axis=1)
        new_data_list.append(batch)

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


# Assuming 'data' is your original DataFrame
model_name = "llama3.2:3b"
final_result = generate_question_answer_pairs(
    data,
    model_name=model_name,
    output_folder="./data_v1/",
    batch_size=1,
    nrows=5,
)
# final_result.to_parquet(f"../data/labeled_jokes_classification_{model_name}.parquet")

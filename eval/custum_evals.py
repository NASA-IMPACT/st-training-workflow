import asyncio
import csv
import heapq
import json
import logging
import os
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from gen_openai_emb import generate_openai_embeddings
from sentence_transformers import util
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    NanoBEIREvaluator,
    TripletEvaluator,
)
from sentence_transformers.evaluation.NanoBEIREvaluator import (
    DatasetNameType,
    dataset_name_to_id,
)
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.util import (
    is_datasets_available,
    pairwise_cos_sim,
    pairwise_dot_score,
    pairwise_euclidean_sim,
    pairwise_manhattan_sim,
)
from torch import Tensor
from tqdm import tqdm, trange

logger = logging.getLogger(__name__)


# Define a dummy placeholder for the model card data attribute
class DummyModelCardData:
    def set_evaluation_metrics(self, *args, **kwargs):
        """This is a dummy method. It does nothing."""
        pass


# Define the complete, self-contained dummy model
class DummyModel:
    """
    A standalone placeholder model that mimics all necessary attributes
    and methods for the InformationRetrievalEvaluator when using
    pre-computed embeddings.
    """

    def __init__(self):
        self.similarity = util.cos_sim
        self.similarity_fn_name = "cosine"
        self.model_card_data = DummyModelCardData()

    def start_multi_process_pool(self, *args, **kwargs):
        """Dummy method. Returns an empty dict."""
        return {}

    def stop_multi_process_pool(self, pool):
        """Dummy method. Does nothing."""
        pass


class MultiGPUInformationRetrievalEvaluator(InformationRetrievalEvaluator):
    def __init__(self, encode_batch_size=128, encode_chunk_size=1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encode_batch_size = encode_batch_size
        self.encode_chunk_size = encode_chunk_size

    # Override the compute_metrices method to customize the evaluation process
    def compute_metrices(
        self,
        model: SentenceTransformer,
        corpus_model=None,
        corpus_embeddings: Tensor | None = None,
        output_path: str | None = None,
        corpus_df: pd.DataFrame | None = None,  # new parameter
        query_df: pd.DataFrame | None = None,  # new parameter
        query_prompt_str: str | None = None,  # new parameter
    ) -> dict[str, float]:
        if corpus_model is None:
            corpus_model = model

        max_k = max(
            max(self.mrr_at_k),
            max(self.ndcg_at_k),
            max(self.accuracy_at_k),
            max(self.precision_recall_at_k),
            max(self.map_at_k),
        )

        pool = model.start_multi_process_pool()
        # Compute embedding for the queries
        if query_df is None:
            logger.info(f"Computing embeddings for queries")
            print("Computing query embeddings")
            query_embeddings = model.encode(
                self.queries,
                pool=pool,
                batch_size=self.encode_batch_size,
                chunk_size=self.encode_chunk_size,
                show_progress_bar=True,
                prompt_name=self.query_prompt_name,
                prompt=query_prompt_str
                if query_prompt_str is not None
                else self.query_prompt,
            )
        else:
            # filter and reorder the embeddings to only include the queries we have
            query_df = query_df.set_index("id").loc[self.queries_ids].reset_index()
            query_embeddings = torch.tensor(query_df["embeddings"].values.tolist())

        queries_result_list = {}
        for name in self.score_functions:
            queries_result_list[name] = [[] for _ in range(len(query_embeddings))]

        if (corpus_embeddings is None) and (corpus_df is None):
            logger.info(f"Computing embeddings for corpus")
            print("Computing corpus embeddings")
            corpus_embeddings = model.encode(
                self.corpus,
                pool=pool,
                batch_size=self.encode_batch_size,
                chunk_size=self.encode_chunk_size,
                show_progress_bar=True,
                prompt_name=self.corpus_prompt_name,
                prompt=self.corpus_prompt,
            )
        elif corpus_df is not None:
            corpus_embeddings = torch.tensor(corpus_df["embeddings"].values.tolist())
        model.stop_multi_process_pool(pool)

        # Iterate over chunks of the corpus
        for corpus_start_idx in trange(
            0,
            len(self.corpus),
            self.corpus_chunk_size,
            desc="Corpus Chunks",
            disable=not self.show_progress_bar,
        ):
            corpus_end_idx = min(
                corpus_start_idx + self.corpus_chunk_size,
                len(self.corpus),
            )

            # Encode chunk of corpus
            if corpus_embeddings is None:
                sub_corpus_embeddings = self.embed_inputs(
                    corpus_model,
                    self.corpus[corpus_start_idx:corpus_end_idx],
                    encode_fn_name="document",
                    prompt_name=self.corpus_prompt_name,
                    prompt=self.corpus_prompt,
                )
            else:
                sub_corpus_embeddings = corpus_embeddings[
                    corpus_start_idx:corpus_end_idx
                ]

            # Compute cosine similarites
            for name, score_function in self.score_functions.items():
                pair_scores = score_function(query_embeddings, sub_corpus_embeddings)

                # Get top-k values
                pair_scores_top_k_values, pair_scores_top_k_idx = torch.topk(
                    pair_scores,
                    min(max_k, len(pair_scores[0])),
                    dim=1,
                    largest=True,
                    sorted=False,
                )
                pair_scores_top_k_values = pair_scores_top_k_values.cpu().tolist()
                pair_scores_top_k_idx = pair_scores_top_k_idx.cpu().tolist()

                for query_itr in range(len(query_embeddings)):
                    for sub_corpus_id, score in zip(
                        pair_scores_top_k_idx[query_itr],
                        pair_scores_top_k_values[query_itr],
                    ):
                        corpus_id = self.corpus_ids[corpus_start_idx + sub_corpus_id]
                        # NOTE: TREC/BEIR/MTEB skips cases where the corpus_id is the same as the query_id, e.g.:
                        # if corpus_id == self.queries_ids[query_itr]:
                        #     continue
                        # This is not done here, as this might be unexpected behaviour if the user just uses
                        # sets of integers from 0 as query_ids and corpus_ids.
                        if len(queries_result_list[name][query_itr]) < max_k:
                            # heaqp tracks the quantity of the first element in the tuple
                            heapq.heappush(
                                queries_result_list[name][query_itr],
                                (score, corpus_id),
                            )
                        else:
                            heapq.heappushpop(
                                queries_result_list[name][query_itr],
                                (score, corpus_id),
                            )

        for name in queries_result_list:
            for query_itr in range(len(queries_result_list[name])):
                for doc_itr in range(len(queries_result_list[name][query_itr])):
                    score, corpus_id = queries_result_list[name][query_itr][doc_itr]
                    queries_result_list[name][query_itr][doc_itr] = {
                        "corpus_id": corpus_id,
                        "score": score,
                    }

        if self.write_predictions and output_path is not None:
            for name in queries_result_list:
                base_filename = self.predictions_file.replace(
                    ".jsonl",
                    f"_{name}.jsonl",
                )
                json_path = os.path.join(output_path, base_filename)
                mode = "w"  # Always create a new file for each score function

                with open(json_path, mode=mode, encoding="utf-8") as fOut:
                    for query_itr in range(len(queries_result_list[name])):
                        query_id = self.queries_ids[query_itr]
                        query_text = self.queries[query_itr]
                        results = queries_result_list[name][query_itr]

                        # Sort results by score in descending order
                        results = sorted(
                            results,
                            key=lambda x: x["score"],
                            reverse=True,
                        )

                        prediction = {
                            "query_id": query_id,
                            "query": query_text,
                            "results": results,
                        }

                        fOut.write(json.dumps(prediction) + "\n")

        logger.info(f"Queries: {len(self.queries)}")
        logger.info(f"Corpus: {len(self.corpus)}\n")

        # Compute scores
        for query_itr in range(len(queries_result_list["cosine"])):
            try:
                query_id = self.queries_ids[query_itr]
            except IndexError:
                print("query_itr: ", query_itr)

        scores = {
            name: self.compute_metrics(queries_result_list[name])
            for name in self.score_functions
        }

        # Output
        for name in self.score_function_names:
            logger.info(f"Score-Function: {name}")
            self.output_scores(scores[name])

        return scores


class MultiGPUTripletEvaluator(TripletEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # overrding the method
    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> dict[str, float]:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"

        logger.info(
            f"TripletEvaluator: Evaluating the model on the {self.name} dataset{out_txt}:",
        )

        pool = model.start_multi_process_pool()
        with nullcontext() if self.truncate_dim is None else model.truncate_sentence_embeddings(
            self.truncate_dim,
        ):
            embeddings_anchors = model.encode(
                self.anchors,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=True,
                pool=pool,
            )
            embeddings_positives = model.encode(
                self.positives,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=True,
                pool=pool,
            )
            embeddings_negatives = model.encode(
                self.negatives,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=True,
                pool=pool,
            )
        model.stop_multi_process_pool(pool)
        if not self.similarity_fn_names:
            self.similarity_fn_names = [model.similarity_fn_name]
            self._append_csv_headers(self.similarity_fn_names)

        similarity_functions = {
            "cosine": lambda anchors, positives, negatives: (
                pairwise_cos_sim(anchors, positives),
                pairwise_cos_sim(anchors, negatives),
            ),
            "dot": lambda anchors, positives, negatives: (
                pairwise_dot_score(anchors, positives),
                pairwise_dot_score(anchors, negatives),
            ),
            "manhattan": lambda anchors, positives, negatives: (
                pairwise_manhattan_sim(anchors, positives),
                pairwise_manhattan_sim(anchors, negatives),
            ),
            "euclidean": lambda anchors, positives, negatives: (
                pairwise_euclidean_sim(anchors, positives),
                pairwise_euclidean_sim(anchors, negatives),
            ),
        }

        metrics = {}
        for fn_name in self.similarity_fn_names:
            if fn_name in similarity_functions:
                positive_scores, negative_scores = similarity_functions[fn_name](
                    embeddings_anchors,
                    embeddings_positives,
                    embeddings_negatives,
                )
                accuracy = (
                    (positive_scores > negative_scores + self.margin[fn_name])
                    .float()
                    .mean()
                    .item()
                )
                metrics[f"{fn_name}_accuracy"] = accuracy
                logger.info(
                    f"Accuracy {fn_name.capitalize()} Similarity:\t{accuracy:.2%}",
                )

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline="", mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps] + list(metrics.values()))

            else:
                with open(csv_path, newline="", mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps] + list(metrics.values()))

        if len(self.similarity_fn_names) > 1:
            metrics["max_accuracy"] = max(metrics.values())

        if self.main_similarity_function:
            self.primary_metric = {
                SimilarityFunction.COSINE: "cosine_accuracy",
                SimilarityFunction.DOT_PRODUCT: "dot_accuracy",
                SimilarityFunction.EUCLIDEAN: "euclidean_accuracy",
                SimilarityFunction.MANHATTAN: "manhattan_accuracy",
            }.get(self.main_similarity_function)
        else:
            if len(self.similarity_fn_names) > 1:
                self.primary_metric = "max_accuracy"
            else:
                self.primary_metric = f"{self.similarity_fn_names[0]}_accuracy"

        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics


class MultiGPUNanoBEIREvaluator(NanoBEIREvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # overrdinnf this method to replace the default InformationRetrievalEvaluator with MultiGPUInformationRetrievalEvaluator
    def _load_dataset(
        self,
        dataset_name: DatasetNameType,
        **ir_evaluator_kwargs,
    ) -> InformationRetrievalEvaluator:
        if not is_datasets_available():
            raise ValueError(
                "datasets is not available. Please install it to use the NanoBEIREvaluator via `pip install datasets`.",
            )
        from datasets import load_dataset

        dataset_path = dataset_name_to_id[dataset_name.lower()]
        corpus = load_dataset(dataset_path, "corpus", split="train")
        queries = load_dataset(dataset_path, "queries", split="train")
        qrels = load_dataset(dataset_path, "qrels", split="train")
        corpus_dict = {
            sample["_id"]: sample["text"]
            for sample in corpus
            if len(sample["text"]) > 0
        }
        queries_dict = {
            sample["_id"]: sample["text"]
            for sample in queries
            if len(sample["text"]) > 0
        }
        qrels_dict = {}
        for sample in qrels:
            if sample["query-id"] not in qrels_dict:
                qrels_dict[sample["query-id"]] = set()
            qrels_dict[sample["query-id"]].add(sample["corpus-id"])

        if self.query_prompts is not None:
            ir_evaluator_kwargs["query_prompt"] = self.query_prompts.get(
                dataset_name,
                None,
            )
        if self.corpus_prompts is not None:
            ir_evaluator_kwargs["corpus_prompt"] = self.corpus_prompts.get(
                dataset_name,
                None,
            )
        human_readable_name = self._get_human_readable_name(dataset_name)
        return MultiGPUInformationRetrievalEvaluator(
            queries=queries_dict,
            corpus=corpus_dict,
            relevant_docs=qrels_dict,
            name=f"nanobeir__{human_readable_name}____evaluator",
            **ir_evaluator_kwargs,
        )

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str = None,
        epoch: int = -1,
        steps: int = -1,
        corpus_dfs: dict = None,
        query_dfs: dict = None,
        *args,
        **kwargs,
    ) -> dict[str, float]:
        per_metric_results = {}
        per_dataset_results = {}
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"
        logger.info(
            f"NanoBEIR Evaluation of the model on {self.dataset_names} dataset{out_txt}:",
        )

        if self.score_functions is None:
            self.score_functions = {model.similarity_fn_name: model.similarity}
            self.score_function_names = [model.similarity_fn_name]
            self._append_csv_headers(self.score_function_names)

        for evaluator in tqdm(
            self.evaluators,
            desc="Evaluating datasets",
            disable=not self.show_progress_bar,
        ):
            print(f"Evaluating {evaluator.name}")
            if corpus_dfs and query_dfs:
                evaluation = evaluator(
                    model,
                    output_path,
                    epoch,
                    steps,
                    corpus_df=corpus_dfs.get(evaluator.name.split("__")[1])
                    if corpus_dfs
                    else None,
                    query_df=query_dfs.get(evaluator.name.split("__")[1])
                    if query_dfs
                    else None,
                )
            else:
                evaluation = evaluator(model, output_path, epoch, steps)
            for k in evaluation:
                if self.truncate_dim:
                    # dataset, _, metric = k.split("_", maxsplit=2)
                    nanobeir_name, dataset_name, subset_name, rest_part = k.split("__")
                    evaluator_name, _, metric = rest_part.split("_", maxsplit=2)
                else:
                    # lets parse the key correctly
                    nanobeir_name, dataset_name, subset_name, rest_part = k.split("__")
                    evaluator_name, metric = rest_part.split("_", maxsplit=1)

                if metric not in per_metric_results:
                    per_metric_results[metric] = []
                # per_dataset_results[dataset + "_" + metric] = evaluation[k]
                per_dataset_results[
                    f"{nanobeir_name}__{dataset_name}__{subset_name}__{evaluator_name}_{metric}"
                ] = evaluation[k]
                per_metric_results[metric].append(evaluation[k])

        agg_results = {}
        for metric in per_metric_results:
            agg_results[metric] = self.aggregate_fn(per_metric_results[metric])

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")

            else:
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data = [epoch, steps]
            for name in self.score_function_names:
                for k in self.accuracy_at_k:
                    output_data.append(agg_results[f"{name}_accuracy@{k}"])

                for k in self.precision_recall_at_k:
                    output_data.append(agg_results[f"{name}_precision@{k}"])
                    output_data.append(agg_results[f"{name}_recall@{k}"])

                for k in self.mrr_at_k:
                    output_data.append(agg_results[f"{name}_mrr@{k}"])

                for k in self.ndcg_at_k:
                    output_data.append(agg_results[f"{name}_ndcg@{k}"])

                for k in self.map_at_k:
                    output_data.append(agg_results[f"{name}_map@{k}"])

            fOut.write(",".join(map(str, output_data)))
            fOut.write("\n")
            fOut.close()

        if not self.primary_metric:
            if self.main_score_function is None:
                score_function = max(
                    [
                        (name, agg_results[f"{name}_ndcg@{max(self.ndcg_at_k)}"])
                        for name in self.score_function_names
                    ],
                    key=lambda x: x[1],
                )[0]
                self.primary_metric = f"{score_function}_ndcg@{max(self.ndcg_at_k)}"
            else:
                self.primary_metric = (
                    f"{self.main_score_function.value}_ndcg@{max(self.ndcg_at_k)}"
                )

        avg_queries = np.mean([len(evaluator.queries) for evaluator in self.evaluators])
        avg_corpus = np.mean([len(evaluator.corpus) for evaluator in self.evaluators])
        logger.info(f"Average Queries: {avg_queries}")
        logger.info(f"Average Corpus: {avg_corpus}\n")

        for name in self.score_function_names:
            logger.info(f"Aggregated for Score Function: {name}")
            for k in self.accuracy_at_k:
                logger.info(
                    "Accuracy@{}: {:.2f}%".format(
                        k,
                        agg_results[f"{name}_accuracy@{k}"] * 100,
                    ),
                )

            for k in self.precision_recall_at_k:
                logger.info(
                    "Precision@{}: {:.2f}%".format(
                        k,
                        agg_results[f"{name}_precision@{k}"] * 100,
                    ),
                )
                logger.info(
                    "Recall@{}: {:.2f}%".format(
                        k,
                        agg_results[f"{name}_recall@{k}"] * 100,
                    ),
                )

            for k in self.mrr_at_k:
                logger.info("MRR@{}: {:.4f}".format(k, agg_results[f"{name}_mrr@{k}"]))

            for k in self.ndcg_at_k:
                logger.info(
                    "NDCG@{}: {:.4f}".format(k, agg_results[f"{name}_ndcg@{k}"]),
                )

        agg_results = self.prefix_name_to_metrics(agg_results, self.name)
        self.store_metrics_in_model_card_data(model, agg_results, epoch, steps)

        per_dataset_results.update(agg_results)

        return per_dataset_results


def get_embedding_for_dataset(
    dataset_config,
    embedding_path,
    dataset_name,
    model_name,
    subset=None,
    data_file=None,
):
    """
    Function to get embeddings for a specific dataset.
    If the embeddings do not exist, it generates them.
    """

    base_path = (
        os.path.join(embedding_path, dataset_name, subset)
        if subset is not None
        else os.path.join(embedding_path, dataset_name)
    )
    # if it is nanobeir then subset is None

    print(f"Base Path: {base_path}")
    corpus_path = os.path.join(base_path, "corpus_embeddings.parquet")
    queries_path = os.path.join(base_path, "queries_embeddings.parquet")

    if (not os.path.exists(corpus_path)) or (not os.path.exists(queries_path)):
        dataset_input_path = None

        # Either path is not None eg. nasa sde v1, nasa sde v2, nasa smd ir (which is hf path)
        if dataset_config.get("path") is not None:
            dataset_input_path = dataset_config["path"]
        # either paths is not None like Nanobeir but path is non
        elif dataset_config.get("paths") is not None:
            # need to be updated
            dataset_input_path = dataset_config[
                "paths"
            ]  # there are multiple paths for different subsets
        # either path needs to be local like beir and also has subsets
        elif dataset_config.get("dataset_cache_path") is not None:
            dataset_input_path = os.path.join(
                dataset_config["dataset_cache_path"],
                subset,
            )

        if isinstance(dataset_input_path, dict):
            # when there is paths
            # genererate embedding for all subsets save it all and return dfs inside a dict
            corpus_dfs = {}
            queries_dfs = {}

            for name, path in dataset_input_path.items():
                corpus_path = os.path.join(base_path, name, "corpus_embeddings.parquet")
                queries_path = os.path.join(
                    base_path,
                    name,
                    "queries_embeddings.parquet",
                )

                if (os.path.exists(corpus_path)) and (os.path.exists(queries_path)):
                    print("Found existing embeddings for", name)
                    corpus_df = pd.read_parquet(corpus_path)
                    queries_df = pd.read_parquet(queries_path)

                else:
                    corpus_df, queries_df = asyncio.run(
                        generate_openai_embeddings(
                            path,
                            os.path.join(base_path, name),
                            model=model_name,
                        ),
                    )

                corpus_dfs[name] = corpus_df
                queries_dfs[name] = queries_df
            return corpus_dfs, queries_dfs
        else:
            corpus_df, queries_df = asyncio.run(
                generate_openai_embeddings(
                    dataset_input_path,
                    base_path,
                    model=model_name,
                ),
            )

            return corpus_df, queries_df

    else:
        corpus_df = pd.read_parquet(corpus_path)
        queries_df = pd.read_parquet(queries_path)
        return corpus_df, queries_df

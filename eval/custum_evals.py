import heapq
import json
import logging
import os

import torch
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    NanoBEIREvaluator,
)
from sentence_transformers.evaluation.NanoBEIREvaluator import (
    DatasetNameType,
    dataset_name_to_id,
)
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.util import is_datasets_available
from torch import Tensor
from tqdm import trange

logger = logging.getLogger(__name__)


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
        query_embeddings: Tensor | None = None,  # new parameter
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
        if query_embeddings is None:
            logger.info(f"Computing embeddings for queries")
            print("Computing query embeddings")
            query_embeddings = model.encode(
                self.queries,
                pool=pool,
                batch_size=self.encode_batch_size,
                chunk_size=self.encode_chunk_size,
                show_progress_bar=True,
                prompt_name=self.query_prompt_name,
                prompt=self.query_prompt,
            )

        queries_result_list = {}
        for name in self.score_functions:
            queries_result_list[name] = [[] for _ in range(len(query_embeddings))]

        if corpus_embeddings is None:
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
        scores = {
            name: self.compute_metrics(queries_result_list[name])
            for name in self.score_functions
        }

        # Output
        for name in self.score_function_names:
            logger.info(f"Score-Function: {name}")
            self.output_scores(scores[name])

        return scores


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
            name=human_readable_name,
            **ir_evaluator_kwargs,
        )

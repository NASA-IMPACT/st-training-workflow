import csv
import json
import os

import datasets

logger = datasets.logging.get_logger(__name__)


_DESCRIPTION = "BEIR Benchmark"
_DATASETS = [
    "msmarco",
    "nfcorpus",
    "fiqa",
    "trec-covid",
    "nq",
    "hotpotqa",
    "arguana",
    "webis-touche2020",
    "cqadupstack",
    "quora",
    "dbpedia-entity",
    "scidocs",
    "fever",
    "climate-fever",
    "scifact",
]

URL_BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/"

# The _URLs dictionary should point to the ZIP file for each dataset
_URLs = {dataset: URL_BASE + f"{dataset}.zip" for dataset in _DATASETS}


class BEIR(datasets.GeneratorBasedBuilder):
    """BEIR BenchmarkDataset."""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=dataset,
            description=f"This is the {dataset} dataset in BEIR Benchmark.",
        )
        for dataset in _DATASETS
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "query": datasets.Value("string"),
                    "relevant": [
                        {
                            "_id": datasets.Value("string"),
                            "score": datasets.Value("int32"),
                        },
                    ],
                },
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        # my_url will now be the URL to the .zip file for the current dataset
        my_url = _URLs[self.config.name]

        # dl_manager.download_and_extract will download the zip and extract it
        # data_dir will be the path to the *extracted folder*
        # (e.g., something like /tmp/datasets/scifact/)
        extracted_data_root = dl_manager.download_and_extract(my_url)

        # Now, construct the paths to the individual files relative to the extracted root
        # The BEIR datasets typically extract to a folder named after the dataset
        # so you might have /tmp/datasets/scifact/scifact/queries.jsonl
        # or just /tmp/datasets/scifact/queries.jsonl

        # Let's assume the extracted content is directly in extracted_data_root,
        # or try checking for a subfolder named after the dataset.
        # A safer way to get the actual data folder within the extracted root:
        actual_data_folder = os.path.join(extracted_data_root, self.config.name)
        if not os.path.exists(actual_data_folder):
            # If the extracted zip doesn't create a subfolder named after the dataset
            # (e.g., if msmarco.zip extracts directly into /tmp/hash/queries.jsonl),
            # then the extracted_data_root is the actual data folder.
            actual_data_folder = extracted_data_root

        # Construct file paths relative to the actual data folder
        queries_path = os.path.join(actual_data_folder, "queries.jsonl")
        qrels_dir = os.path.join(actual_data_folder, "qrels")
        qrels_paths = {
            "train": os.path.join(qrels_dir, "train.tsv"),
            "dev": os.path.join(qrels_dir, "dev.tsv"),
            "test": os.path.join(qrels_dir, "test.tsv"),
        }

        # All train, dev and test splits available for these datasets
        if self.config.name in ["msmarco", "nfcorpus", "hotpotqa", "fiqa", "fever"]:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "query_path": queries_path,
                        "qrels_path": qrels_paths["train"],
                    },
                ),
                datasets.SplitGenerator(
                    name="dev",
                    gen_kwargs={
                        "query_path": queries_path,
                        "qrels_path": qrels_paths["dev"],
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "query_path": queries_path,
                        "qrels_path": qrels_paths["test"],
                    },
                ),
            ]

        # Only train and test splits available for these datasets
        elif self.config.name in ["nq", "scifact"]:
            # No need to pop from my_urls anymore as we're managing paths after extraction
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "query_path": queries_path,
                        "qrels_path": qrels_paths["train"],
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "query_path": queries_path,
                        "qrels_path": qrels_paths["test"],
                    },
                ),
            ]

        # Only dev and test splits available for these datasets
        elif self.config.name in [
            "dbpedia-entity",
            "quora",
        ]:  # Changed "dbpedia" to "dbpedia-entity" to match _DATASETS
            return [
                datasets.SplitGenerator(
                    name="dev",
                    gen_kwargs={
                        "query_path": queries_path,
                        "qrels_path": qrels_paths["dev"],
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "query_path": queries_path,
                        "qrels_path": qrels_paths["test"],
                    },
                ),
            ]

        # Only test split available for these datasets
        else:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "query_path": queries_path,
                        "qrels_path": qrels_paths["test"],
                    },
                ),
            ]

    def _generate_examples(self, query_path, qrels_path):
        """Yields examples."""

        queries, qrels = {}, {}

        # Check if query_path exists before trying to open
        if not os.path.exists(query_path):
            logger.warning(f"Query file not found at: {query_path}")
            return  # Skip if query file doesn't exist

        with open(query_path, encoding="utf-8") as fIn:
            text = fIn.readlines()

        for line in text:
            line = json.loads(line)
            queries[line.get("_id")] = line.get("text", "")

        # Check if qrels_path exists before trying to open
        if not os.path.exists(qrels_path):
            logger.warning(f"Qrels file not found at: {qrels_path}")
            return  # Skip if qrels file doesn't exist

        reader = csv.reader(
            open(qrels_path, encoding="utf-8"),
            delimiter="\t",
            quoting=csv.QUOTE_MINIMAL,
        )

        next(reader)  # Skip header

        for id, row in enumerate(reader):
            query_id, corpus_id, score = row[0], row[1], int(row[2])
            if query_id not in qrels:
                qrels[query_id] = {corpus_id: score}
            else:
                qrels[query_id][corpus_id] = score

        for i, query_id in enumerate(qrels):
            yield i, {
                "query": queries.get(
                    query_id,
                    "",
                ),  # Use .get to handle cases where query_id might not be in queries (unlikely but safer)
                "relevant": [
                    {
                        "_id": doc_id,
                        "score": score,
                    }
                    for doc_id, score in qrels[query_id].items()
                ],
            }

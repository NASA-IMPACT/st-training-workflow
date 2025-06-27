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

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/"
_URLs = {
    dataset: {
        "queries": URL + f"{dataset}/queries.jsonl",
        "qrels": {
            "train": URL + f"{dataset}/qrels/train.tsv",
            "dev": URL + f"{dataset}/qrels/dev.tsv",
            "test": URL + f"{dataset}/qrels/test.tsv",
        },
    }
    for dataset in _DATASETS
}


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

        my_urls = _URLs[self.config.name]

        # All train, dev and test splits available for these datasets
        if self.config.name in ["msmarco", "nfcorpus", "hotpotqa", "fiqa", "fever"]:
            data_dir = dl_manager.download_and_extract(my_urls)
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "query_path": data_dir["queries"],
                        "qrels_path": data_dir["qrels"]["train"],
                    },
                ),
                datasets.SplitGenerator(
                    name="dev",
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "query_path": data_dir["queries"],
                        "qrels_path": data_dir["qrels"]["dev"],
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "query_path": data_dir["queries"],
                        "qrels_path": data_dir["qrels"]["test"],
                    },
                ),
            ]

        # Only train and test splits available for these datasets
        elif self.config.name in ["nq", "scifact"]:
            my_urls["qrels"].pop("dev", None)
            data_dir = dl_manager.download_and_extract(my_urls)

            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "query_path": data_dir["queries"],
                        "qrels_path": data_dir["qrels"]["train"],
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "query_path": data_dir["queries"],
                        "qrels_path": data_dir["qrels"]["test"],
                    },
                ),
            ]

        # Only dev and test splits available for these datasets
        elif self.config.name in ["dbpedia", "quora"]:
            my_urls["qrels"].pop("train", None)
            data_dir = dl_manager.download_and_extract(my_urls)
            return [
                datasets.SplitGenerator(
                    name="dev",
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "query_path": data_dir["queries"],
                        "qrels_path": data_dir["qrels"]["dev"],
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "query_path": data_dir["queries"],
                        "qrels_path": data_dir["qrels"]["test"],
                    },
                ),
            ]

        # Only test split available for these datasets
        else:
            for split in ["train", "dev"]:
                my_urls["qrels"].pop(split, None)
            data_dir = dl_manager.download_and_extract(my_urls)
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "query_path": data_dir["queries"],
                        "qrels_path": data_dir["qrels"]["test"],
                    },
                ),
            ]

    def _generate_examples(self, query_path, qrels_path):
        """Yields examples."""

        queries, qrels = {}, {}

        with open(query_path, encoding="utf-8") as fIn:
            text = fIn.readlines()

        for line in text:
            line = json.loads(line)
            queries[line.get("_id")] = line.get("text", "")

        reader = csv.reader(
            open(qrels_path, encoding="utf-8"),
            delimiter="\t",
            quoting=csv.QUOTE_MINIMAL,
        )

        next(reader)

        for id, row in enumerate(reader):
            query_id, corpus_id, score = row[0], row[1], int(row[2])
            if query_id not in qrels:
                qrels[query_id] = {corpus_id: score}
            else:
                qrels[query_id][corpus_id] = score

        for i, query_id in enumerate(qrels):
            yield i, {
                "query": queries[query_id],
                "relevant": [
                    {
                        "_id": doc_id,
                        "score": score,
                    }
                    for doc_id, score in qrels[query_id].items()
                ],
            }

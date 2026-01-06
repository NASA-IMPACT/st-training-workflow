import argparse
import json
import os
import pathlib
import time
from collections import defaultdict
from string import Template

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns
import torch
from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader
from custum_evals import (
    DummyModel,
    MultiGPUInformationRetrievalEvaluator,
    MultiGPUNanoBEIREvaluator,
    UBinarySentenceTransformer,
    get_embedding_for_dataset,
    hamming_similarity_from_distance,
)
from datasets import load_dataset
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sentence_transformers import models as s_models

load_dotenv()
import sys

sys.path.append("../model_exploration/")  # Add training dir to path

parser = argparse.ArgumentParser(description="Sentence Transformer Training Config")

parser.add_argument(
    "--dataset_name",
    type=str,
    default=None,
    choices=[
        "nanobeir",
        "beir",
        "nasa_sde_ir_v1",
        "nasa_sde_ir_v2",
        "nasa_sde_ir_v3",
        "nasa_sde_ir_20251024_v5",
        "nasa_sde_ir_v4",
        "nasa_smd_ir",
        "shortform-fullform",
        "nasa_repo_code_benchmark_v0.1",
        "codesearchnet_testset_benchmark_v0.2",
        "codesearchnet_testset_benchmark_v0.1",
        "code_repo_search_benchmark_v1"
    ],
)
parser.add_argument("--ks", nargs="*", default=[1, 3, 5, 10])
parser.add_argument("--plotks", nargs="*", default=[1, 3, 5, 10])
parser.add_argument("--json_output_path", type=str, default="results_json/")
parser.add_argument("--output_dir_plots", type=str, default="results_plots/")
parser.add_argument("--json_time_path", type=str, default="results_times/")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument(
    "--just_plot",
    type=int,
    default=0,
    choices=[0, 1],
    help="Set to 1 to just plot the results without running the evaluation.",
)
parser.add_argument(
    "--desired_metric_types",
    nargs="+",
    default=["mrr"],
    help="A list of metrics to plot (e.g., mrr, accuracy, ndcg, precision, recall, map).",
)


args = parser.parse_args()

dataset_name = args.dataset_name
ks = args.ks
plotks = [int(k) for k in args.plotks]
json_output_path = args.json_output_path
json_time_path = args.json_time_path
output_dir_plots = args.output_dir_plots
batch_size = args.batch_size
just_plot = args.just_plot
desired_metric_types = args.desired_metric_types


os.makedirs(json_output_path, exist_ok=True)
os.makedirs(json_time_path, exist_ok=True)
os.makedirs(output_dir_plots, exist_ok=True)
json_time_path = f"{json_time_path}/{dataset_name}_corpus_embedding_times.json"
json_output_path = f"{json_output_path}/{dataset_name}_eval_dump.json"


similarity_fns = {
    "hamming": hamming_similarity_from_distance,
}

models = {
    "modernbert-embed-base": {
        "path": "nomic-ai/modernbert-embed-base",
        "color": "#1f77b4",
        "hatch": "//",  # Diagonal lines
        "model_config": {
            "torch_dtype": torch.float16,
        },
        # "display_name": "ModernBERT",
    },
    "nasa-smd-ibm-st-v2": {
        "path": "nasa-impact/nasa-smd-ibm-st-v2",
        "color": "#ff7f0e",
        "hatch": "\\\\",  # Reverse diagonal lines
        "display_name": "Original INDUS-ST",
    },
    # "indus-sde-st-v0.1": {"path": "nasa-impact/indus-sde-st-v0.1", "color": "#2ca02c"},
    # "indus-sde-st-v0.2_whole-moon-14": {
    #     "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/"
    #     "model-qr3ln5om:v1/checkpoint-116000",
    #     "color": "#d62728",
    # },
    # "indus-sde-st-v0.2_atomic-plasma-15": {
    #     "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/"
    #     "model-ykf0bews:v1/checkpoint-13000",
    #     "color": "#9467bd",
    # },
    # "indus-sde-st-v0.2_vocal-river-16": {
    #     "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/"
    #     "model-dexwvlkj:v1/checkpoint-13000",
    #     "color": "#8c564b",
    # },
    # "indus-sde-st-v0.2_super-armadillo-25": {
    #     "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/"
    #     "model-1ogtkl75:v1/checkpoint-268500",
    #     "color": "#e377c2",
    # },
    # "indus-sde-st-v0.2_drawn-puddle-31": {
    #     "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/"
    #     "model-3x845j4d:v1/checkpoint-11000",
    #     "color": "#7f7f7f",
    # },
    # "indus-sde-st-v0.2_peach-night-57": {
    #     "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/"
    #     "model-rpv7vnpd:v1/checkpoint-13500",
    #     "color": "#bcbd22",
    # },
    # "indus-sde-st-v0.2_peach-night-57_42k": {
    #     "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/model-rpv7vnpd:v1/checkpoint-42000",
    #     "color": "#2affdb",
    # },
    # "indus-sde-st-v0.2_polar-monkey-61_20k": {
    #     "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/model-6hjbp1bx:v1/checkpoint-20000",
    #     "color": "#ffbb78",
    # },
    "indus-sde-st-v0.2_polar-monkey-61_30k": {
        "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/model-6hjbp1bx:v1/checkpoint-30000",
        "color": "#33ff77",
        "hatch": "...",  # Dots
        "display_name": "INDUS-SDE",
    },
    "indus-sde-st-v0.2_polar-monkey-61_30k-512t": {
        "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/model-6hjbp1bx:v1/checkpoint-30000",
        "color": "#ccff33",
        "hatch": "...",  # Dots
        "display_name": "INDUS-SDE-512-token",
        "max_seq_length": 512,
    },
    # "nasa-smd-ibm-st-v2(ft_ads_sde)": {
    #     "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/model-xfpc778s:v1/checkpoint-1492",
    #     "color": "#6622ee",
    # },
    # "deploy_model_v2": {
    #     "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/deploy_model_v2",
    #     "color": "#6e9944",
    # },
    # "deployed_model_v1_slow_token_cls_pool": {
    #     "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/deploy_model_v1",
    #     "color": "#cc4444",
    #     "pooling_mode": "cls",  # This is the special flag
    # },
    # "Qwen3-Embedding-0.6B": {
    #     "path": "Qwen/Qwen3-Embedding-0.6B",
    #     "color": "#bcbd22",
    #     "query_prompt": "Instruct: Given a search query (could be a question, title, or text),"
    #     " retrieve relevant scientific passages that answer or describe the query. \nQuery:",
    #     }
    "indus-sde-st-v0.2-61_30k-ubinary_emb": {
        "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/model-6hjbp1bx:v1/checkpoint-30000",
        "color": "#9933aa",
        "hatch": "xxx",
        "similarity_fn_name": "hamming",
        "display_name": "INDUS-SDE-post-binarization",
    },
    # "s2_azure-eon-73": {
    #     "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/s2_binarized_model/azure-eon-73/checkpoint-32500",
    #     "color": "#00a9ac",
    #     "similarity_fn_name": "hamming",
    # },
    # "s2_cosmic-pine-77": {
    #     "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/s2_binarized_model/cosmic-pine-77/checkpoint-1458",
    #     "color": "#008dad",
    #     "similarity_fn_name": "hamming",
    # },
    # "s2_swift-morning-78": {
    #     "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/s2_binarized_model/swift-morning-78/checkpoint-1458",
    #     "color": "#005197",
    #     "similarity_fn_name": "hamming",
    # },
    # "s2_robust-firebrand-83": {
    #     "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/s2_binarized_model/robust-firebrand-83/checkpoint-15000",
    #     "color": "#af38f5",
    #     "similarity_fn_name": "hamming",
    # },
    # "s2_celestial-butterfly-84": {
    #     "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/s2_binarized_model/celestial-butterfly-84/checkpoint-15000",
    #     "color": "#0d06ac",
    #     "similarity_fn_name": "hamming",
    # },
    # "s2_absurd-snowflake-85": {
    #     "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/s2_binarized_model/absurd-snowflake-85/checkpoint-20000",
    #     "color": "#005282",
    #     "similarity_fn_name": "hamming",
    # },
    "s2_ruby-water-86": {
        "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/s2_binarized_model/ruby-water-86/checkpoint-20000",
        "color": "#669933",
        "hatch": "+.",
        "similarity_fn_name": "hamming",
        "display_name": "s2-86-20k",
    },
    "s2_ruby-water-86-30k": {
        "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/s2_binarized_model/ruby-water-86/checkpoint-30000",
        "color": "#644690",
        "hatch": "..",
        "similarity_fn_name": "hamming",
        "display_name": "s2-86-30k",
    },
    "s2_ruby-water-86-38k": {
        "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/s2_binarized_model/ruby-water-86/checkpoint-38500",
        "color": "#7B995D",
        "hatch": "--",
        "similarity_fn_name": "hamming",
        "display_name": "s2-86-38k",
    },
    # "s2_azure-serenity-87": {
    #     "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/s2_binarized_model/azure-serenity-87/checkpoint-10000",
    #     "color": "#669999",
    #     "similarity_fn_name": "hamming",
    # },
    # "s2_rose-smoke-88": {
    #     "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/s2_binarized_model/rose-smoke-88/checkpoint-10000",
    #     "color": "#992211",
    #     "similarity_fn_name": "hamming",
    # },
    # "s2_restful-bird-93": {
    #     "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/s2_binarized_model/restful-bird-93/checkpoint-20000",
    #     "color": "#997511",
    #     "similarity_fn_name": "hamming",
    #     "comment": "QAT: SDE weighted 41/2"
    # },
    # "s2_eternal-energy-94": {
    #     "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/s2_binarized_model/eternal-energy-94/checkpoint-27500",
    #     "color": "#119985",
    #     "similarity_fn_name": "hamming",
    #     "comment": "QAT: Without SDE dataset (Just to test the performance)"
    # },
    # "s2_worthy-dust-96": {
    #     "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/s2_binarized_model/worthy-dust-96/checkpoint-15500",
    #     "color": "#667788",
    #     "similarity_fn_name": "hamming",
    #     "comment": "QAT: All the stage 2 datasets are having equal weights. = 1"
    # },
    # "s2_wandering-snowflake-98": {
    #     "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/s2_binarized_model/wandering-snowflake-98/checkpoint-20000",
    #     "color": "#92A447",
    #     "similarity_fn_name": "hamming",
    #     "comment": "QAT: Uses weight_decay = 0.1"
    # }

    "granite-embedding-small-english-r2": {
        "path": "ibm-granite/granite-embedding-small-english-r2",
        "color": "#bcbd22",
        "hatch": "..-",
        "model_config": {
            "torch_dtype": torch.float16,
        },
    },
    "granite-embedding-small-english-r2-512t": {
        "path": "ibm-granite/granite-embedding-small-english-r2",
        "color": "#bd8422",
        "hatch": "o--",
        "model_config": {
            "torch_dtype": torch.float16,
        },
        "max_seq_length": 512,
        "display_name": "granite-small-english-r2-512-token",
    },
    "granite-embedding-small-english-r2-1024t": {
        "path": "ibm-granite/granite-embedding-small-english-r2",
        "color": "#bd7722",
        "hatch": "+--",
        "model_config": {
            "torch_dtype": torch.float16,
        },
        "max_seq_length": 1024,
        "display_name": "granite-small-english-r2-1024-token",
    },
    # "granite-embedding-english-r2": {
    #     "path": "ibm-granite/granite-embedding-english-r2",
    #     "color": "#6e9944"
    # },
    # "s3_firm-dust-7": {
    #     "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/stage3_models/firm-dust-7/checkpoint-12782",
    #     "color": "#ff4a0e",
    # },
    # "s3_twilight-forest-14_3k(granite_small_r2)": {
    #     "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/"
    #     "stage3_models/twilight-forest-14/checkpoint-3000",
    #     "color": "#036b71",
    #     "model_config": {
    #         "torch_dtype": torch.float16,
    #     },
    # },
    # "s3_twilight-forest-14_102k(granite_small_r2)": {
    #     "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/stage3_models/twilight-forest-14/checkpoint-102257",
    #     "color": "#136b71",
    #     "model_config": {
    #         "torch_dtype": torch.float16,
    #     },
    # },
    # "s3_amber-thunder-15": {
    #     "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/artifacts/stage3_models/amber-thunder-15/checkpoint-12991",
    #     "color": "#32490e",
    # },
    # "codebert-base": {
    #     "path": "microsoft/codebert-base",
    #     "color": "#9933aa",
    # },
    # "CodeBERTa-small-v1": {
    #     "path": "huggingface/CodeBERTa-small-v1",
    #     "color": "#1166cc",
    # },
    # "jina-code-embeddings-0.5b": {
    #     "path": "jinaai/jina-code-embeddings-0.5b",
    #     "color": "#CB1B88",
    #     "model_config": {
    #         "torch_dtype": torch.float16
    #     }
    # }
}

embeddings = {
    # "[OpenAI]text-embedding-3-small": {
    #     "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/openai_emb_cache/"
    #     "text-embedding-3-small/",
    #     "model_name": "text-embedding-3-small",
    #     "color": "#17becf",
    # },
    # "[OpenAI]text-embedding-3-large": {
    #     "path": "/rhome/sawale/indus_traning/sentense_transformers/eval/openai_emb_cache/"
    #     "text-embedding-3-large/",
    #     "model_name": "text-embedding-3-large",
    #     "color": "#393b79",
    # },
}

dataset_config = {
    "nanobeir": {
        "path": None,
        "subsets": [None],  # this means to use all datasets
        "paths": {
            "NanoClimateFEVER": "zeta-alpha-ai/NanoClimateFEVER",
            "NanoDBPedia": "zeta-alpha-ai/NanoDBPedia",
            "NanoFEVER": "zeta-alpha-ai/NanoFEVER",
            "NanoFiQA2018": "zeta-alpha-ai/NanoFiQA2018",
            "NanoHotpotQA": "zeta-alpha-ai/NanoHotpotQA",
            "NanoMSMARCO": "zeta-alpha-ai/NanoMSMARCO",
            "NanoNFCorpus": "zeta-alpha-ai/NanoNFCorpus",
            "NanoNQ": "zeta-alpha-ai/NanoNQ",
            "NanoQuoraRetrieval": "zeta-alpha-ai/NanoQuoraRetrieval",
            "NanoSCIDOCS": "zeta-alpha-ai/NanoSCIDOCS",  # issue
            "NanoArguAna": "zeta-alpha-ai/NanoArguAna",
            "NanoSciFact": "zeta-alpha-ai/NanoSciFact",
            "NanoTouche2020": "zeta-alpha-ai/NanoTouche2020",
        },
    },
    "beir": {
        "path": None,
        "subsets": [
            "trec-covid",
            "nfcorpus",
            "nq",
            "hotpotqa",
            "fiqa",
            "arguana",
            "webis-touche2020",
            "dbpedia-entity",
            "scidocs",
            "fever",
            "climate-fever",
            "scifact",
        ],
        "dataset_cache_path": "./beir_datasets",
    },
    "nasa_sde_ir_v1": {"path": "nasa-impact/nasa-sde-IR-benchmark-sample-v1"},
    "nasa_sde_ir_v2": {"path": "nasa-impact/nasa-sde-IR-benchmark-sample-v2"},
    "nasa_smd_ir": {"path": "nasa-impact/nasa-smd-IR-benchmark"},
    "nasa_sde_ir_v3": {
        "path": "nasa-impact/nasa-sde-IR-benchmark-sample-v3",
        "data_files": [
            "qrels/question-answer~SDE_general_v2.tsv",
            "qrels/question-answer~SDE_general_v3.tsv",
            "qrels/search_term-document~CMR.tsv",
            "qrels/search_term-document~PDS.tsv",
            "qrels/search_term-document~SDE_general_v2.tsv",
            "qrels/search_term-document~SDE_general_v3.tsv",
            "qrels/title-description~CMR.tsv",
            "qrels/title-description~PDS.tsv",
        ],
        "data_files_colors": [
            "#1f77b4",  # question-answer~SDE_general_v2.tsv
            "#ff7f0e",  # question-answer~SDE_general_v3.tsv
            "#2ca02c",  # search_term-document~CMR.tsv
            "#d62728",  # search_term-document~PDS.tsv
            "#9467bd",  # search_term-document~SDE_general_v2.tsv
            "#8c564b",  # search_term-document~SDE_general_v3.tsv
            "#e377c2",  # title-description~CMR.tsv
            "#7f7f7f",  # title-description~PDS.tsv
        ],
    },
    "nasa_sde_ir_20251024_v5": {
        "path": "nasa-impact/nasa-sde-IR-benchmark-20251024-v5",
        "data_files": [
            "qrels/qa_pairs.tsv",
            "qrels/search_pairs.tsv",
        ],
        "data_files_colors": [
            "#1f77b4",  # qa_pairs.tsv
            "#ff7f0e",  # search_pairs.tsv
        ]
    },
    "code_repo_search_benchmark_v1": {
        "path": "nasa-impact/code_repo_search_benchmark_v1",
        "data_files": [
            "qrels/astro.tsv",
            "qrels/earth.tsv",
            "qrels/planetary.tsv"
        ],
        "data_files_colors": [
            "#1f77b4",  # astro.tsv
            "#ff7f0e",  # earth.tsv
            "#2ca02c",  # planetary.tsv
        ]
    },
    "nasa_sde_ir_v4": {
        "path": "nasa-impact/nasa-sde-IR-benchmark-sample-v4",
        "data_files": [
            "qrels/search_term-document~CMR.tsv",
            "qrels/search_term-document~PDS.tsv",
            "qrels/search_term-document~SDE_general_v2.tsv",
            "qrels/search_term-document~SDE_general_v3.tsv",
            "qrels/title-description~CMR.tsv",
            "qrels/title-description~PDS.tsv",
        ],
        "data_files_colors": [
            "#2ca02c",  # search_term-document~CMR.tsv
            "#d62728",  # search_term-document~PDS.tsv
            "#9467bd",  # search_term-document~SDE_general_v2.tsv
            "#8c564b",  # search_term-document~SDE_general_v3.tsv
            "#e377c2",  # title-description~CMR.tsv
            "#7f7f7f",  # title-description~PDS.tsv
        ],
    },
    "shortform-fullform": {
        "path": "/rhome/sawale/indus_traning/sentense_transformers/data/short_full_form_pairs",
        # "path": "/rhome/sawale/indus_traning/sentense_transformers/data/short_full_form_pairs_v1",
        "data_files": [
            "qrels/chrono_units.tsv",
            "qrels/data_format.tsv",
            "qrels/instruments.tsv",
            "qrels/locations.tsv",
            "qrels/measurement_name.tsv",
            "qrels/mime_type.tsv",
            "qrels/platforms.tsv",
            "qrels/projects.tsv",
            "qrels/providers.tsv",
            "qrels/ru_content_type.tsv",
            "qrels/sciencekeywords.tsv",
            "qrels/temporal_resolution_range.tsv",
            "qrels/pim_astronomy_and_astrophysics_flight_missions.tsv",
            "qrels/pim_beyond_earth_missions.tsv",
            "qrels/pim_ceos_instruments.tsv",
            "qrels/pim_ceos_missions.tsv",
            "qrels/pim_gcmd_instruments.tsv",
            "qrels/pim_gcmd_platforms.tsv",
            "qrels/pim_high_energy_astrophysics_missions.tsv",
            "qrels/pim_mast_missions.tsv",
            "qrels/pim_nasa_heliophysics_sun-planet_missions.tsv",
            "qrels/pim_pds_mission_archive_page.tsv",
            "qrels/pim_planetary_missions_beyond_earth_orbit.tsv",
            "qrels/pim_spase_instruments.tsv",
            "qrels/pim_spase_observatories.tsv",
        ],
        "data_files_colors": [
            "#1f77b4",  # qrels/chrono_units.tsv
            "#ff7f0e",  # qrels/data_format.tsv
            "#2ca02c",  # qrels/instruments.tsv
            "#d62728",  # qrels/locations.tsv
            "#9467bd",  # qrels/measurement_name.tsv
            "#8c564b",  # qrels/mime_type.tsv
            "#e377c2",  # qrels/platforms.tsv
            "#7f7f7f",  # qrels/projects.tsv
            "#bcbd22",  # qrels/providers.tsv
            "#17becf",  # qrels/ru_content_type.tsv
            "#aec7e8",  # qrels/sciencekeywords.tsv
            "#ffbb78",  # qrels/temporal_resolution_range.tsv
            "#98df8a",  # qrels/pim_astronomy_and_astrophysics_flight_missions.tsv
            "#c5b0d5",  # qrels/pim_beyond_earth_missions.tsv
            "#c49c94",  # qrels/pim_ceos_instruments.tsv
            "#f7b6d2",  # qrels/pim_ceos_missions.tsv
            "#c7c7c7",  # qrels/pim_gcmd_instruments.tsv
            "#dbdb8d",  # qrels/pim_gcmd_platforms.tsv
            "#9edae5",  # qrels/pim_high_energy_astrophysics_missions.tsv
            "#6b6ecf",  # qrels/pim_mast_missions.tsv
            "#9c9ede",  # qrels/pim_nasa_heliophysics_sun-planet_missions.tsv
            "#bd9e39",  # qrels/pim_pds_mission_archive_page.tsv
            "#e7ba52",  # qrels/pim_planetary_missions_beyond_earth_orbit.tsv
            "#e7cb94",  # qrels/pim_spase_instruments.tsv
            "#843c39",  # qrels/pim_spase_observatories.tsv
        ],
    },
    "nasa_repo_code_benchmark_v0.1": {
        "path": "nasa-impact/nasa_repo_code_benchmark_v0.1",
        "data_files": [
            "qrels/nasa_science_class_code_docstring_heldout.tsv",
            "qrels/nasa_science_class_code_identifier_heldout.tsv",
            "qrels/nasa_science_function_code_docstring_heldout.tsv",
            "qrels/nasa_science_function_code_identifier_heldout.tsv",
            "qrels/python.tsv",
            "qrels/java.tsv",
            "qrels/javascript.tsv",
            "qrels/c.tsv",
            "qrels/c++.tsv",
            "qrels/fortran.tsv",
            "qrels/matlab.tsv",
        ],
        "data_files_colors": [
            "#1f77b4",  # nasa_science_class_code_docstring_heldout.tsv
            "#ff7f0e",  # nasa_science_class_code_identifier_heldout.tsv
            "#2ca02c",  # nasa_science_function_code_docstring_heldout.tsv
            "#d62728",  # nasa_science_function_code_identifier_heldout.tsv
            "#9467bd",  # python.tsv
            "#8c564b",  # java.tsv
            "#e377c2",  # javascript.tsv
            "#7f7f7f",  # c.tsv
            "#bcbd22",  # c++.tsv
            "#17becf",  # fortran.tsv
            "#aec7e8",  # matlab.tsv
        ],
    },
    "codesearchnet_testset_benchmark_v0.2": {
        "path": "nasa-impact/codesearchnet_testset_benchmark_v0.2",
        "data_files": [
            "qrels/python.tsv",
            "qrels/java.tsv",
            "qrels/javascript.tsv",
            "qrels/php.tsv",
            "qrels/ruby.tsv",
            "qrels/go.tsv",
        ],
        "data_files_colors": [
            "#1f77b4",  # qrels/python.tsv
            "#ff7f0e",  # qrels/java.tsv
            "#2ca02c",  # qrels/javascript.tsv
            "#d62728",  # qrels/php.tsv
            "#9467bd",  # qrels/ruby.tsv
            "#8c564b",  # qrels/go.tsv
        ],
    },
    "codesearchnet_testset_benchmark_v0.1": {
        "path": "nasa-impact/codesearchnet_testset_benchmark_v0.1",
        "data_files": [
            "qrels/python.tsv",
            "qrels/java.tsv",
            "qrels/javascript.tsv",
            "qrels/php.tsv",
            "qrels/ruby.tsv",
            "qrels/go.tsv",
        ],
        "data_files_colors": [
            "#1f77b4",  # qrels/python.tsv
            "#ff7f0e",  # qrels/java.tsv
            "#2ca02c",  # qrels/javascript.tsv
            "#d62728",  # qrels/php.tsv
            "#9467bd",  # qrels/ruby.tsv
            "#8c564b",  # qrels/go.tsv
        ],
    },
}


def dataset_getter(
    dataset_name,
    corpus_split="train",
    queries_split="train",
    relevant_docs_split="test",
    data_file=None,
):
    corpus = load_dataset(
        dataset_config[dataset_name]["path"],
        data_files="corpus.jsonl",
        split=corpus_split,
        token=os.environ["HUGGINGFACE_TOKEN"],
    )
    queries = load_dataset(
        dataset_config[dataset_name]["path"],
        data_files="queries.jsonl",
        split=queries_split,
        token=os.environ["HUGGINGFACE_TOKEN"],
    )
    relevant_docs_data = load_dataset(
        dataset_config[dataset_name]["path"],
        split=relevant_docs_split,
        data_files=data_file,
        token=os.environ["HUGGINGFACE_TOKEN"],
    )

    corpus = {row["_id"]: row["text"] for i, row in enumerate(corpus)}
    queries = {row["_id"]: row["text"] for row in queries}
    relevant_docs_data = (
        relevant_docs_data.to_pandas()
        .groupby("query-id")["corpus-id"]
        .apply(set)
        .to_dict()
    )
    relevant_docs_data = {
        str(k): {str(item) for item in v} for k, v in relevant_docs_data.items()
    }

    return corpus, queries, relevant_docs_data


def beir_dataset_getter(subset):
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{subset}.zip"
    out_dir = os.path.join(
        pathlib.Path.cwd(),
        dataset_config["beir"]["dataset_cache_path"],
    )
    data_path = beir_util.download_and_unzip(url, out_dir)

    # Load the dataset using the BEIR loader
    corpus_beir, queries, qrels_beir = GenericDataLoader(data_folder=data_path).load(
        split="test",
    )

    # The corpus needs to be flattened from Dict[str, Dict[str,str]] to Dict[str, str]
    # We'll concatenate the title and text for each document.
    corpus = {
        doc_id: (doc.get("title", "") + " " + doc.get("text", "")).strip()
        for doc_id, doc in corpus_beir.items()
    }

    # The qrels need to be converted to the relevant_docs format: Dict[str, Set[str]]
    # We'll only consider documents with a relevance score > 0.
    relevant_docs = defaultdict(set)
    for query_id, doc_scores in qrels_beir.items():
        for doc_id, score in doc_scores.items():
            if score > 0:
                relevant_docs[query_id].add(doc_id)

    return corpus, queries, relevant_docs


def get_dataset(dataset_name, subset=None, relevant_docs_split="test", data_file=None):
    if dataset_name.lower() in ["beir"]:
        return beir_dataset_getter(subset)
    elif dataset_name.lower() in ["nanobeir"]:
        return None, None, None
    else:
        return dataset_getter(
            dataset_name,
            relevant_docs_split=relevant_docs_split,
            data_file=data_file,
        )


def get_evaluator(
    dataset_name: str,
    queries: dict,
    corpus: dict,
    relevant_docs_data: dict,
    subset=None,
    data_file=None,
):

    if dataset_name.lower() == "nanobeir":
        args = dict(
            dataset_names=None,
            mrr_at_k=ks,
            accuracy_at_k=ks,
            precision_recall_at_k=ks,
            map_at_k=ks,
            ndcg_at_k=ks,
            show_progress_bar=True,
            batch_size=batch_size,
            write_csv=True,
        )
        evaluators = {
            **{"cosine": MultiGPUNanoBEIREvaluator(**args)},
            **{
                name: MultiGPUNanoBEIREvaluator(**args, score_functions={name: fn})
                for name, fn in similarity_fns.items()
            },
        }

    elif dataset_name.lower() == "beir":
        args = dict(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs_data,
            name=f"beir__{subset}____evaluator",
            batch_size=batch_size,
            mrr_at_k=ks,
            ndcg_at_k=ks,
            accuracy_at_k=ks,
            precision_recall_at_k=ks,
            map_at_k=ks,
            show_progress_bar=True,
            write_csv=True,
            encode_chunk_size=5000,
            encode_batch_size=batch_size,
        )
        evaluators = {
            **{"cosine": MultiGPUInformationRetrievalEvaluator(**args)},
            **{
                name: MultiGPUInformationRetrievalEvaluator(
                    **args,
                    score_functions={name: fn},
                )
                for name, fn in similarity_fns.items()
            },
        }

    elif dataset_name.lower() == "nasa_sde_ir_v1":
        args = dict(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs_data,
            name=f"{dataset_name}______evaluator",
            batch_size=batch_size,
            mrr_at_k=ks,
            ndcg_at_k=ks,
            accuracy_at_k=ks,
            precision_recall_at_k=ks,
            map_at_k=ks,
            show_progress_bar=True,
            write_csv=True,
            encode_chunk_size=5000,
            encode_batch_size=batch_size,
        )
        evaluators = {
            **{"cosine": MultiGPUInformationRetrievalEvaluator(**args)},
            **{
                name: MultiGPUInformationRetrievalEvaluator(
                    **args,
                    score_functions={name: fn},
                )
                for name, fn in similarity_fns.items()
            },
        }

    elif dataset_name.lower() == "nasa_sde_ir_v2":
        args = dict(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs_data,
            name=f"{dataset_name}______evaluator",
            batch_size=batch_size,
            mrr_at_k=ks,
            ndcg_at_k=ks,
            accuracy_at_k=ks,
            precision_recall_at_k=ks,
            map_at_k=ks,
            show_progress_bar=True,
            write_csv=True,
            encode_chunk_size=5000,
            encode_batch_size=batch_size,
        )
        evaluators = {
            **{"cosine": MultiGPUInformationRetrievalEvaluator(**args)},
            **{
                name: MultiGPUInformationRetrievalEvaluator(
                    **args,
                    score_functions={name: fn},
                )
                for name, fn in similarity_fns.items()
            },
        }

    elif dataset_name.lower() in [
        "nasa_sde_ir_v3",
        "nasa_sde_ir_20251024_v5",
        "shortform-fullform",
        "nasa_sde_ir_v4",
        "nasa_repo_code_benchmark_v0.1",
        "nasa_repo_code_benchmark_v0.2",
        "nasa_repo_code_benchmark_v0.3",
        "codesearchnet_testset_benchmark_v0.1",
        "codesearchnet_testset_benchmark_v0.2",
        "code_repo_search_benchmark_v1"
    ]:
        args = dict(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs_data,
            name=f"{dataset_name}____{data_file}__evaluator",
            batch_size=batch_size,
            mrr_at_k=ks,
            ndcg_at_k=ks,
            accuracy_at_k=ks,
            precision_recall_at_k=ks,
            map_at_k=ks,
            show_progress_bar=True,
            write_csv=True,
            encode_chunk_size=1000,
            encode_batch_size=batch_size,
            corpus_chunk_size=500,
        )
        evaluators = {
            **{"cosine": MultiGPUInformationRetrievalEvaluator(**args)},
            **{
                name: MultiGPUInformationRetrievalEvaluator(
                    **args,
                    score_functions={name: fn},
                )
                for name, fn in similarity_fns.items()
            },
        }

    elif dataset_name.lower() == "nasa_smd_ir":
        args = dict(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs_data,
            name=f"{dataset_name}______evaluator",
            batch_size=batch_size,
            mrr_at_k=ks,
            ndcg_at_k=ks,
            accuracy_at_k=ks,
            precision_recall_at_k=ks,
            map_at_k=ks,
            show_progress_bar=True,
            write_csv=True,
            encode_chunk_size=5000,
            encode_batch_size=batch_size,
        )
        evaluators = {
            **{"cosine": MultiGPUInformationRetrievalEvaluator(**args)},
            **{
                name: MultiGPUInformationRetrievalEvaluator(
                    **args,
                    score_functions={name: fn},
                )
                for name, fn in similarity_fns.items()
            },
        }

    return evaluators


# ...existing code...
def add_mean_metrics(all_results, query_counts, mean_basis="subset"):
    global models
    for model_name in all_results:

        similarity_fn_names = set(
            [i.split("_")[-2] for i in all_results[model_name].keys()],
        )
        if len(similarity_fn_names) > 1:
            raise ValueError(
                f"Multiple similarity functions found for {model_name}: "
                f"{similarity_fn_names}. Please handle manually. Skipping...",
            )

        similarity_fn_name = similarity_fn_names.pop()
        metric_names = set(
            [i.split("_")[-1] for i in list(all_results[model_name].keys())],
        )

        if mean_basis == "subset":
            mean_basis_names = set(
                [
                    i.split("__")[1]
                    for i in all_results[model_name]
                    if "mean" not in i.split("__")[1]
                ],
            )
            key_name = "${dataset_name}__${mean_basis_name}____evaluator_${similarity_fn_name}_${metric}"
            result_key_name = "${dataset_name}__${mean_type}____evaluator_${similarity_fn_name}_${metric}"
        elif mean_basis == "data_file":
            mean_basis_names = set(
                [
                    i.split("__")[2]
                    for i in all_results[model_name]
                    if "mean" not in i.split("__")[2]
                ],
            )
            key_name = "${dataset_name}____${mean_basis_name}__evaluator_${similarity_fn_name}_${metric}"
            result_key_name = "${dataset_name}____${mean_type}__evaluator_${similarity_fn_name}_${metric}"

        mean_result = {}
        weighted_mean_result = {}

        def _get_weight_for(mean_name: str) -> int:
            """
            Query_counts keys are created in evaluate() as:
              f'{dataset_name}__{subset if subset is not None else ""}__{data_file if data_file is not None else ""}'
            For subset means we match keys that start with 'dataset__subset__'
            For data_file means we match keys that end with '__data_file' (or contain it as the last segment).
            Fall back to 1 if no matching entry found.
            """
            # exact match
            exact = query_counts.get(mean_name)
            if exact is not None:
                return exact
            # try matching prefixes / suffixes
            ds_prefix = f"{dataset_name}__{mean_name}__"
            total = 0
            found = False
            for k, v in query_counts.items():
                if k.startswith(ds_prefix):
                    total += v
                    found = True
                elif k.endswith(f"__{mean_name}") or k.split("__")[-1] == mean_name:
                    total += v
                    found = True
            return total if found else 1

        # Calculate the mean for each metric
        for metric in metric_names:
            values = []
            weighted_values = []
            weights = []
            for mean_basis_name in mean_basis_names:
                _key_name = Template(key_name).substitute(
                    dataset_name=dataset_name,
                    mean_basis_name=mean_basis_name,
                    similarity_fn_name=similarity_fn_name,
                    metric=metric,
                )
                try:
                    values.append(all_results[model_name][_key_name])
                except KeyError:
                    print(
                        f"Key {_key_name} not found in results for model {model_name}. "
                        f"Found only {list(all_results[model_name].keys())}. Skipping...",
                    )
                    continue

                # compute a sensible weight by matching query_counts keys
                weight = _get_weight_for(mean_basis_name)
                weighted_values.append(values[-1] * weight)
                weights.append(weight)

            _result_key_name = Template(result_key_name).substitute(
                dataset_name=dataset_name,
                mean_type="mean",
                similarity_fn_name=similarity_fn_name,
                metric=metric,
            )
            mean_result[_result_key_name] = sum(
                values,
            ) / (len(values) if len(values) > 0 else 1)

            if sum(weights) > 0:
                __result_key_name = Template(result_key_name).substitute(
                    dataset_name=dataset_name,
                    mean_type="weightedmean",
                    similarity_fn_name=similarity_fn_name,
                    metric=metric,
                )
                weighted_mean_result[__result_key_name] = sum(
                    weighted_values,
                ) / sum(weights)
            else:
                weighted_mean_result[__result_key_name] = 0.0

        all_results[model_name] = {
            **all_results[model_name],
            **mean_result,
            **weighted_mean_result,
        }


def check_if_eval_already_exists(all_results, model_name, subset=None, data_file=None):
    if model_name not in all_results:
        return False

    # get set of all data_files for the model_name
    existing_data_files = set(
        [k.split("__")[2] for k in all_results.get(model_name, {}).keys()],
    )
    if data_file is not None and data_file not in existing_data_files:
        return False

    # similarly get all the subsets for the model_name
    existing_subsets = set(
        [k.split("__")[1] for k in all_results.get(model_name, {}).keys()],
    )
    if subset is not None and subset not in existing_subsets:
        return False

    return True


def pre_compute_corpus_embedding(
    models,
    dataset_name,
    subset,
    all_results,
    dataset_config,
    time_taken,
):
    if dataset_name.lower() in ["nanobeir"]:
        print(
            f"Skipping pre-computation of corpus embeddings for {dataset_name} as it is not supported.",
        )
        return {}, {}
    data_file = dataset_config[dataset_name].get("data_files", [None])
    # n_data_files = len(data_file)

    relevant_docs_split = "train" if any([i is not None for i in data_file]) else "test"
    corpus, _q, _ = get_dataset(
        dataset_name,
        subset,
        relevant_docs_split=relevant_docs_split,
        data_file=data_file[0] if data_file[0] is not None else None,
    )

    corpus_texts = list(corpus.values())
    corpus_pre_computed_embeddings = {}
    for model_name, model_info in models.items():
        if check_if_eval_already_exists(all_results, model_name, subset, data_file[0]):
            print(
                f"Model {model_name} with the subset {subset} and data_file {data_file[0]} "
                f"already evaluated. Skipping...Preembedding of corpus",
            )
            continue

        print(
            f"Pre-computing corpus embeddings for model: {model_name}, subset: {subset}",
        )
        if model_name not in time_taken:
            time_taken[model_name] = {}
        model = load_model_with_proper_pooling(model_name, model_info)

        start_time = time.time()
        pool = model.start_multi_process_pool()

        corpus_embeddings = model.encode(
            corpus_texts,
            pool=pool,
            batch_size=batch_size,
            chunk_size=1000,
            show_progress_bar=True,
        )

        model.stop_multi_process_pool(pool)
        end_time = time.time()
        time_taken[model_name][f"{dataset_name}__{subset}"] = end_time - start_time

        corpus_pre_computed_embeddings[
            f"{dataset_name}__{subset}__{model_name}"
        ] = corpus_embeddings

    return corpus_pre_computed_embeddings, time_taken


def load_json_if_exists(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def generate_query_counts_for_nanobeir(dataset_config, dataset_name):
    query_counts = {}

    for d_name, path in dataset_config[dataset_name]["paths"].items():
        relevant_docs_data = load_dataset(
            path,
            split="train",
            name="qrels",
            token=os.environ["HUGGINGFACE_TOKEN"],
        )
        query_counts[f"{dataset_name}__{d_name}__"] = len(relevant_docs_data)

    return query_counts


def load_model_with_proper_pooling(model_name, model_info):
    if "pooling_mode" in model_info:
        print(
            f"Loading {model_name} with custom '{model_info['pooling_mode']}' pooling...",
        )
        # 1. Load the base transformer model
        transformer_layer = s_models.Transformer(model_info["path"])
        # 2. Create a pooling layer with the specified mode
        pooling_layer = s_models.Pooling(
            word_embedding_dimension=transformer_layer.get_word_embedding_dimension(),
            pooling_mode=model_info["pooling_mode"],  # Use the mode from our config
        )

        base_model = SentenceTransformer(modules=[transformer_layer, pooling_layer])

        # 3. Create the final SentenceTransformer model from these two modules
        if model_info.get("similarity_fn_name") == "hamming":
            print(f"Using UBinarySentenceTransformer for {model_name}")
            model = UBinarySentenceTransformer(modules=base_model._modules.values())
        else:
            print(f"Using SentenceTransformer for {model_name}")
            model = base_model

    else:
        base_model = SentenceTransformer(
            model_info["path"],
            model_kwargs=model_info.get("model_config", {}),
        )
        # This is the default behavior for all other models
        print(f"Loading {model_name} with default pooling...")
        if model_info.get("similarity_fn_name") == "hamming":
            print(f"Using UBinarySentenceTransformer for {model_name}")
            model = UBinarySentenceTransformer(
                modules=base_model._modules.values(),
            )
        else:
            print(f"Using SentenceTransformer for {model_name}")
            model = base_model

    if model_info.get("max_seq_length") is not None:
        print(
            f"Setting max_seq_length to {model_info['max_seq_length']} for model {model_name}",
        )
        model.max_seq_length = int(model_info["max_seq_length"])
    return model


def evaluate():
    time_taken = load_json_if_exists(json_time_path)
    all_results = load_json_if_exists(json_output_path)
    query_counts = {}

    subsets = dataset_config[dataset_name].get("subsets", [None])
    for subset in subsets:
        # this will loop multiple times if subsets are provided else it will loop once
        # check if there is multiple data_files for the dataset_name

        # precompute corpus embeddings for different dataset_name-subset-model_name
        # for different data_files, only relevant_docs / qrels are different
        corpus_pre_computed_embeddings, time_taken = pre_compute_corpus_embedding(
            models,
            dataset_name,
            subset,
            all_results,
            dataset_config,
            time_taken,
        )
        for data_file in dataset_config[dataset_name].get("data_files", [None]):
            corpus, queries, relevant_docs = get_dataset(
                dataset_name,
                subset,
                relevant_docs_split="test" if data_file is None else "train",
                data_file=data_file,
            )
            if relevant_docs is not None:
                query_counts[
                    f"{dataset_name}__{subset if subset is not None else ''}__"
                    f"{data_file if data_file is not None else ''}"
                ] = len(relevant_docs)
            evaluators = get_evaluator(
                dataset_name,
                queries,
                corpus,
                relevant_docs,
                subset,
                data_file,
            )
            # Looping models
            for model_name, model_info in models.items():
                print(
                    f"Evaluating model: {model_name}, subset {subset} and data_file: {data_file}",
                )
                if check_if_eval_already_exists(
                    all_results,
                    model_name,
                    subset,
                    data_file,
                ):
                    print(
                        f"Model {model_name} with the subset {subset} and data_file "
                        f"{data_file} already evaluated. Skipping...",
                    )
                    continue
                model = load_model_with_proper_pooling(model_name, model_info)
                results = evaluators.get(
                    model_info.get("similarity_fn_name", "cosine"),
                )(
                    model,
                    query_prompt_str=model_info.get("query_prompt", None),
                    corpus_embeddings=corpus_pre_computed_embeddings.get(
                        f"{dataset_name}__{subset}__{model_name}",
                    ),
                )
                results = {
                    k: v for k, v in results.items() if k.startswith(dataset_name)
                }  # filtering out non compatible keys
                if model_name not in all_results:
                    all_results[model_name] = {}
                all_results[model_name] = {**all_results[model_name], **results}

            # Looping through the embeddings
            for embedding_name, embedding_info in embeddings.items():
                print(
                    f"Evaluating embedding: {embedding_name}, "
                    f"subset {subset} and data_file: {data_file}",
                )
                if check_if_eval_already_exists(
                    all_results,
                    embedding_name,
                    subset,
                    data_file,
                ):
                    print(
                        f"Embedding {embedding_name} already evaluated in json. Skipping...",
                    )
                    continue

                print(
                    f"Loading embeddings for {embedding_name} from {embedding_info['path']}",
                )
                # check if the embedding for the dataset_name exists if not generate it: calling a function
                corpus_df, queries_df = get_embedding_for_dataset(
                    dataset_config=dataset_config[dataset_name],
                    embedding_path=embedding_info["path"],
                    dataset_name=dataset_name,
                    model_name=embedding_info["model_name"],
                    subset=subset,
                    data_file=data_file,
                )

                dummy_model = DummyModel()

                if isinstance(corpus_df, pd.DataFrame) and isinstance(
                    queries_df,
                    pd.DataFrame,
                ):
                    results = evaluators.get(
                        model_info.get("similarity_fn_name", "cosine"),
                    )(
                        model=dummy_model,
                        corpus_df=corpus_df,
                        query_df=queries_df,
                    )
                elif isinstance(corpus_df, dict) and isinstance(queries_df, dict):
                    results = evaluators.get(
                        model_info.get("similarity_fn_name", "cosine"),
                    )(
                        model=dummy_model,
                        corpus_dfs=corpus_df,
                        query_dfs=queries_df,
                    )
                    # filteriing the results to only include the valid keys
                    results = {
                        k: v for k, v in results.items() if k.startswith(dataset_name)
                    }

                if embedding_name not in all_results:
                    all_results[embedding_name] = {}
                all_results[embedding_name] = {**all_results[embedding_name], **results}

    if len(dataset_config[dataset_name].get("paths", [])) > 1:
        # computing query counts for different paths
        query_counts = generate_query_counts_for_nanobeir(dataset_config, dataset_name)

    if len(subsets) > 1 or len(dataset_config[dataset_name].get("paths", [])) > 1:
        # need to add a mean of metrics from different subsets of different models
        add_mean_metrics(all_results, query_counts, mean_basis="subset")
    elif len(dataset_config[dataset_name].get("data_files", [])) > 1:
        # need to add a mean of metrics from different data_files of different models
        add_mean_metrics(all_results, query_counts, mean_basis="data_file")

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    with open(json_time_path, "w", encoding="utf-8") as f:
        json.dump(time_taken, f, ensure_ascii=False, indent=4)


def convert_json_output_to_df(json_output_path):
    if not os.path.exists(json_output_path):
        print(
            f"JSON output path {json_output_path} does not exist."
            "Please run the evaluation first.",
        )
        return
    with open(json_output_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    # lets make a df first of the json
    records = []
    for model, metrics in all_results.items():
        for metric_name_full, value in metrics.items():
            parts = metric_name_full.split("__")
            dataset_name_str = parts[0]
            subset_str = parts[1]
            data_file_str = parts[2]
            remaining_str = parts[-1]

            parts = remaining_str.split("@")
            k = int(parts[1])
            metric_name = parts[0].split("_")[-1]
            records.append(
                {
                    "dataset_name": dataset_name_str,
                    "subset": subset_str,
                    "data_file": data_file_str,
                    "model": model,
                    "metric": metric_name,
                    "k": k,
                    "value": value,
                },
            )

    # Create a pandas DataFrame
    df = pd.DataFrame(records)

    return df


def plot_results(
    json_output_path, 
    top_ks=[1, 3, 5, 10], 
    dpi=300, 
    legend_cols=2,
    model_name_mapper=None  # <--- Added argument
):
    """
    Plots bar charts for model performance metrics.
    
    Args:
        model_name_mapper (dict, optional): Dictionary mapping internal model names 
                                            to display names. Defaults to None.
    
    Note: This function assumes the following variables are defined in the global scope:
      - desired_metric_types (list)
      - models (dict)
      - embeddings (dict)
      - dataset_name (str)
      - output_dir_plots (str)
      - convert_json_output_to_df (function)
    """
    
    # 0. Set Style for "Beautiful" look
    sns.set_theme(style="whitegrid", rc={"axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.6})

    print(f"Plotting results for K values: {top_ks}...")
    
    # 1. Load & Filter Data
    df = convert_json_output_to_df(json_output_path)
    df = df[df["metric"].isin(desired_metric_types)]
    df = df[df["k"].isin(top_ks)]

    if df.empty:
        print(f"No data found for K values: {top_ks}.")
        return

    all_model_configs = {**models, **embeddings}
    model_color_palette = {name: config["color"] for name, config in all_model_configs.items()}
    model_hatch_palette = {name: config.get("hatch", None) for name, config in all_model_configs.items()}
    
    present_models = set(df["model"].unique())
    sorted_model_names = sorted([m for m in all_model_configs.keys() if m in present_models])
    
    for subset in df["subset"].unique():
        subset_df = df[df["subset"] == subset]

        for data_file in subset_df["data_file"].unique():
            subset_data_file_df = subset_df[subset_df["data_file"] == data_file]

            # === GENERATE PLOT ===
            # We use the original names here to maintain color/hatch mapping consistency
            g = sns.catplot(
                data=subset_data_file_df,
                x="k", y="value", hue="model",
                hue_order=sorted_model_names, 
                col="metric", kind="bar",
                col_wrap=2, sharey=False,
                legend_out=True, palette=model_color_palette,
                height=5,
                aspect=3.5,
                edgecolor="black",     
                linewidth=0.8
            )

            # Apply Hatches
            for ax in g.axes.flat:
                for i, container in enumerate(ax.containers):
                    if i < len(sorted_model_names):
                        model_name = sorted_model_names[i]
                        hatch = model_hatch_palette.get(model_name)
                        if hatch:
                            for bar in container:
                                bar.set_hatch(hatch)
                                # Draw hatch lines in white by using white edgecolor
                                bar.set_edgecolor("white")
                                # Slightly widen edge so hatch is visible
                                bar.set_linewidth(0.9)
                        else:
                            # keep a visible border for non-hatched bars
                            for bar in container:
                                bar.set_edgecolor("black")
                                bar.set_linewidth(0.8)

            # Titles and Labels
            g.set_titles("{col_name}")  
            g.set_axis_labels("K Value", "Score")
            g.despine(left=True, bottom=True)

            # Annotations (Values on top of bars)
            for ax in g.axes.flat:
                for p in ax.patches:
                    if hasattr(p, 'get_height') and p.get_height() > 0:
                        ax.annotate(
                            f"{p.get_height():.2f}", 
                            (p.get_x() + p.get_width() / 2, p.get_height()), 
                            ha="center", va="bottom", xytext=(0, 2), 
                            textcoords="offset points", fontsize=9, fontweight='bold'
                        )

            # === CUSTOM LEGEND HANDLES ===
            legend_handles = []
            for m in sorted_model_names:
                display_label = model_name_mapper.get(m, m) if model_name_mapper else m
                
                edge_color = "white" if model_hatch_palette.get(m) else "black"
                line_width = 0.9 if model_hatch_palette.get(m) else 0.5

                handle = Patch(
                    facecolor=model_color_palette.get(m),
                    hatch=model_hatch_palette.get(m),
                    label=display_label, 
                    edgecolor=edge_color,
                    linewidth=line_width
                )
                legend_handles.append(handle)

            # Clear default legends
            if g.legend: g.legend.remove()
            if g.fig.legends: 
                for legend in g.fig.legends: legend.remove()


            # === FIX 1: Define Layout Variables ===
            # We define these explicitly so we can use them to calculate the center
            plt_left = 0.1
            plt_right = 0.95
            plt_bottom = 0.15
            plt_top = 0.85

            g.fig.subplots_adjust(
                bottom=plt_bottom, 
                top=plt_top, 
                left=plt_left, 
                right=plt_right
            )



            # === FIX 2: Legend Centering ===
            # Instead of magic number 0.3, we anchor to the exact visual center (0.5)
            # loc='upper center' means the Top-Middle of the legend box attaches to the anchor point
            g.fig.legend(
                handles=legend_handles,
                loc="upper center",       
                bbox_to_anchor=(0.3, 0.02), # x=0.2 (center), y=0.02 (very bottom)
                ncol=legend_cols,
                frameon=False,
                fontsize=10,
                columnspacing=1.5        
            )
            

            title = f"Model Performance on {dataset_name} benchmark"
            if subset and subset != "None":
                title += f" - {subset}"
            if data_file and data_file not in [None, "None", ""]:
                # append just the file basename for readability
                title += f" - {os.path.basename(data_file)}"

            # === FIX 3: Calculate Title Center ===
            # The visual center is the midpoint between the left and right margins
            # visual_center_x = (plt_left + plt_right) / 2

            g.fig.suptitle(
                title,
                fontsize=15,
                y=0.96,
                x=0.3,
                fontweight="bold",
                ha="center",
            )

            # Save
            os.makedirs(os.path.join(output_dir_plots, dataset_name), exist_ok=True)
            plt.savefig(
                os.path.join(
                    output_dir_plots, dataset_name,
                    f"{dataset_name}_{subset}_{data_file.split('/')[-1] if data_file else 'all'}.png"
                ),
                bbox_inches="tight", 
                dpi=dpi
            )
            plt.close(g.fig)

def plot_data_files_based_eval(json_output_path, top_ks=[1, 3, 5, 10]):
    print("Plotting data files based evaluation...")
    df = convert_json_output_to_df(json_output_path)

    # filter the DataFrame to include only the desired metric types
    df = df[df["metric"].isin(desired_metric_types)]

    # ~ removing data_file which is mean
    df = df[~df["data_file"].str.contains("mean")]

    if df["data_file"].nunique() <= 1:
        # there is no multiple data files to plot
        return

    output_dir_plots_ = os.path.join(
        output_dir_plots,
        dataset_name,
        "data_files_based_eval",
    )
    os.makedirs(output_dir_plots_, exist_ok=True)

    dataset_color_palette = {
        name: color
        for name, color in zip(
            dataset_config[dataset_name].get("data_files", [None]),
            dataset_config[dataset_name].get("data_files_colors", [None]),
        )
    }
    # Add hatch patterns for data files too
    # dataset_hatch_palette = {
    #     name: hatch
    #     for name, hatch in zip(
    #         dataset_config[dataset_name].get("data_files", [None]),
    #         dataset_config[dataset_name].get("data_files_hatches", [None] * len(dataset_config[dataset_name].get("data_files", []))),
    #     )
    #     }
    # loop through differnt models: each model will have its own plot
    for model_name in df["model"].unique():
        df_model = df[df["model"] == model_name]

        sorted_data_file_names = sorted(df_model["data_file"].unique())

        # Create the bar plot
        g = sns.catplot(
            data=df_model,
            x="k",
            y="value",
            hue="data_file",
            hue_order=sorted_data_file_names,
            col="metric",
            kind="bar",
            col_wrap=2,
            sharey=False,
            height=5,
            aspect=2,
            legend_out=True,
            palette=dataset_color_palette,
        )

        # Customize subplot titles and labels
        g.set_titles("Metric: {col_name}")
        g.set_axis_labels("K Value", "Score")
        g.despine(left=True)

        # Add value labels on top of each bar
        for ax in g.axes.flat:
            for p in ax.patches:
                value = f"{p.get_height():.2f}"
                x = p.get_x() + p.get_width() / 2
                y = p.get_height()
                ax.annotate(
                    value,
                    (x, y),
                    ha="center",
                    va="center",
                    xytext=(0, 5),
                    textcoords="offset points",
                    fontsize=9,
                )

        # 1. Move the legend to be centered below the plot
        sns.move_legend(
            g,
            "lower center",
            bbox_to_anchor=(
                0.5,
                -0.2,
            ),  # Center the legend horizontally, move it down
            ncol=6,  # Adjust number of columns to fit your models (image has 10)
            title="Data Files",
            frameon=False,
        )

        # 2. Add the main title for the figure
        title = f"Performance of {model_name} at K-Value On {dataset_name}"
        g.fig.suptitle(
            title,
            fontsize=16,  # Optional: Adjust font size
        )

        # 3. Use tight_layout to automatically adjust spacing and center the title
        # The rect parameter makes space for the suptitle at the top
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # 4. Save the figure
        # The bbox_inches="tight" argument is crucial for including the legend
        plt.savefig(
            f"{output_dir_plots_}/{model_name}_performance_plots.png",
            bbox_inches="tight",
            dpi=300,  # Optional: Increase image resolution
        )


if __name__ == "__main__":
    if not just_plot:
        evaluate()

    model_name_mapper = {
        model_name: model_info["display_name"]
        for model_name, model_info in models.items()
        if "display_name" in model_info
    }

    plot_results(json_output_path, plotks, model_name_mapper=model_name_mapper, legend_cols=3)
    plot_data_files_based_eval(json_output_path, plotks)
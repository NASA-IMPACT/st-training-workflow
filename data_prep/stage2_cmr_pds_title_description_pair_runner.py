# %% [markdown]
# # Generalized Data Processing and Structuring for SDE Dumps (with Caching)
#
# This script provides a structured and reusable pipeline for processing data dumps
# from the SDE index API. It handles two main types of data:
# 1.  **PDS (Planetary Data System)**
# 2.  **CMR (Common Metadata Repository)**
#
# The pipeline performs the following steps:
# 1.  **Configuration**: Defines all necessary parameters, including file paths,
#     column mappings, and data transformation rules.
# 2.  **Data Loading**: Loads the raw CSV dump, applies human-readable column aliases,
#     and splits the data into PDS and CMR DataFrames.
# 3.  **CMR-Specific Processing (with Caching)**: Fetches supplementary metadata for CMR records from
#     external XML URLs concurrently. It checks for a local cache file first to avoid
#     re-fetching data on subsequent runs.
# 4.  **Data Transformation**: Converts both PDS and CMR DataFrames into a
#     standardized format with `query`, `context`, and `metadata` columns.
# 5.  **Output**: Saves the final processed datasets as Parquet files.

# %%
# %reload_ext autoreload
# %autoreload 2

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import xmltodict
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm
from urllib3.util.retry import Retry

# --- 1. Configuration ---
# Centralize all settings for easy modification.


class Config:
    """Holds all configuration parameters for the data processing pipeline."""

    # --- File Paths ---
    # NOTE: Update these paths to your local environment.
    BASE_DATA_PATH = "/rhome/sawale/indus_traning/sentense_transformers/data/api_data"
    OUTPUT_PATH = "/rhome/sawale/indus_traning/sentense_transformers/data/stage2_sde"

    SOURCE_CSV_PATH = os.path.join(
        BASE_DATA_PATH,
        "processed_dev_sde_index_api_dump_2025-06-19.csv",
    )

    # --- Caching Configuration ---
    # This path stores the result of the time-consuming CMR URL fetching.
    CMR_CACHE_PATH = os.path.join(BASE_DATA_PATH, "cmr_with_purposes.parquet")

    # --- Output Paths ---
    CMR_OUTPUT_PATH = os.path.join(OUTPUT_PATH, "cmr_pairs_structured.parquet")
    PDS_OUTPUT_PATH = os.path.join(OUTPUT_PATH, "pds_pairs_structured.parquet")

    # --- Column Alias Mapping ---
    COLUMN_ALIAS_MAP = {
        "sourcestr1": "agency",
        "sourcestr2": "repository",
        "sourcestr3": "mission",
        "sourcestr4": "dataformat",
        "sourcestr5": "data_process_level",
        "sourcestr6": "data_process_level_desc",
        "sourcestr7": "spatial_bounds_coord_sys",
        "sourcestr12": "urls",
        "sourcestr13": "scientific_focus",
        "sourcestr14": "operation_wav_spectral",
        "sourcestr15": "data_product_desc",
        "sourcestr16": "investigation",
        "sourcestr17": "platform",
        "sourcestr19": "persistent_id",
        "sourcestr20": "instrument",
        "sourcevarchar2": "spatial_bounds",
        "sourcestr21": "spatial_bounds_1",
        "sourcestr22": "spatial_bounds_2",
        "sourcevarchar10": "temporal_bounds",
        "sourcedatetime1": "temporal_bounds_1",
        "sourcedatetime2": "temporal_bounds_2",
        "sourcedatetime3": "temporal_bounds_3",
        "sourcedatetime4": "temporal_bounds_4",
        "sourcedatetime5": "temporal_bounds_5",
        "sourcevarchar5": "spatial_resolution_spatial_extent",
        "sourcevarchar6": "spatial_resolution_spatial_info",
        "sourcestr23": "spatial_resolution_spatial_info_1",
        "sourcestr24": "spatial_resolution_spatial_info_2",
        "sourcestr25": "spatial_resolution_spatial_info_3",
        "sourcestr26": "spatial_resolution_spatial_info_4",
        "sourcestr27": "spatial_resolution_spatial_info_5",
        "sourcestr28": "spatial_resolution_spatial_info_6",
        "sourcestr29": "data_item_title",
        "sourcestr30": "temporal_resolution_temp_keywords",
        "sourcevarchar7": "temporal_resolution_temp_extents",
        "sourcestr31": "temporal_resolution_1",
        "sourcestr32": "temporal_resolution_2",
        "sourcestr33": "temporal_resolution_3",
        "sourcestr34": "temporal_resolution_4",
        "sourcestr35": "temporal_resolution_5",
        "sourcestr36": "data_prod_version",
        "sourcestr37": "pds_bundle",
        "sourcestr38": "pds_collection",
        "sourcestr39": "spase_observed_region",
        "sourcestr40": "pds_lid",
        "sourcestr41": "version_num",
        "sourcestr42": "version_desc",
        "sourcestr43": "ivo_id",
        "sourcestr44": "cap_type",
        "sourcestr45": "ej_desc_simp",
        "sourcestr46": "ej_geographic_coverage",
        "sourcestr47": "ej_latency",
        "sourcestr48": "ej_project",
        "sourcestr49": "ej_strengths",
        "sourcestr50": "ej_limitations",
        "sourcestr51": "ej_data_visualization",
        "sourcestr52": "ej_intended_use",
        "sourcecsv14": "ej_indicators",
        "sourcecsv5": "scientific_keywords_1",
        "sourcecsv6": "scientific_keywords_2",
        "sourcecsv7": "scientific_keywords_3",
        "sourcecsv8": "scientific_keywords_4",
        "sourcecsv9": "scientific_keywords_5",
        "sourcecsv10": "scientific_keywords_6",
        "sourcecsv11": "scientific_keywords_7",
        "sourcecsv12": "cmr_phenomena",
        "sourcecsv13": "cmr_measurement_tech",
        "sourcevarchar11": "data_volume",
        "sourcevarchar12": "access_constraints",
        "sourcevarchar13": "use_constraints",
        "sourcevarchar14": "spase_parameter",
        "sourcevarchar15": "repos_links",
        "sourcecsv1": "related_urls_1",
        "sourcecsv2": "related_urls_2",
        "sourcecsv3": "related_urls_3",
        "sourcecsv4": "related_urls_4",
        "sourcebool1": "IsMetadataViewer",
        "url1": "download_url",
        "sourcestr8": "bps_osdr_id",
        "sourcestr9": "bps_factor",
        "sourcestr10": "bps_organism",
        "sourcestr11": "bps_funding",
        "url2": "ej_sde_link",
        "sourcecsv15": "coord",
        "sourcestr58": "coord_unit",
        "sourcedouble1": "mean_wavel",
        "sourcestr59": "wavel_unit",
        "sourcestr60": "frm_name",
        "sourcebool2": "h_flag",
        "sourcestr61": "pub_date",
        "sourcestr62": "contact",
    }

    # --- PDS Configuration ---
    PDS_RAW_COLS = [
        "id",
        "sourcestr13",
        "sourcecsv5",
        "sourcestr15",
        "sourcestr16",
        "sourcestr17",
        "sourcestr20",
        "title",
        "treepath",
        "sourcestr40",
        "sourcestr37",
        "sourcestr2",
    ]
    PDS_CONTEXT_MAP = {
        "data_product_desc": "Description",
        "scientific_keywords_1": "Target",
        "repository": "Repository",
        "investigation": "Investigation",
        "instrument": "Instrument",
    }
    PDS_METADATA_COLS = [
        "id",
        "pds_lid",
        "treepath",
        "title",
        "data_product_desc",
        "scientific_keywords_1",
        "repository",
        "instrument",
        "platform",
        "investigation",
    ]

    # --- CMR Configuration ---
    CMR_RAW_COLS = [
        "id",
        "url1",
        "treepath",
        "title",
        "sourcestr1",
        "sourcestr15",
        "sourcestr13",
        "sourcecsv5",
        "sourcestr17",
        "sourcestr20",
        "sourcecsv12",
        "sourcestr4",
        "purposes",
    ]
    CMR_CONTEXT_MAP = {
        "data_product_desc": "Description",
        "purposes": "Purpose",
        "scientific_keywords_1": "Scientific Keywords",
        "cmr_phenomena": "CMR Phenomena",
        "dataformat": "Dataformat",
        "instrument": "Instrument",
    }
    CMR_METADATA_COLS = [
        "id",
        "treepath",
        "download_url",
        "title",
        "agency",
        "data_product_desc",
        "scientific_keywords_1",
        "platform",
        "instrument",
        "cmr_phenomena",
        "dataformat",
        "purposes",
    ]


# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# --- 2. Data Loading & Preparation ---


def load_and_split_data(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the source CSV, and splits it into CMR and PDS dataframes.

    Args:
        path: The file path to the source CSV dump.

    Returns:
        A tuple containing the CMR DataFrame and the PDS DataFrame.
    """
    logging.info(f"Loading data from {path}...")
    df = pd.read_csv(path)
    logging.info(f"Loaded {len(df)} total records.")

    # Split data based on the 'treepath' column
    cmr_df = df[df["treepath"] == "/Earth Science/Earth Science Data (CMR)/"].copy()
    pds_df = df[df["treepath"].str.contains("PDS", na=False)].copy()

    logging.info(f"Found {len(cmr_df)} CMR records and {len(pds_df)} PDS records.")
    return cmr_df, pds_df


# --- 3. CMR-Specific Processing ---


class CmrUrlProcessor:
    """
    A class to handle fetching and parsing of metadata from CMR XML URLs.
    """

    def __init__(self, retries: int = 3, backoff_factor: float = 0.5):
        self.session = self._get_session_with_retries(retries, backoff_factor)

    def _get_session_with_retries(
        self,
        retries: int,
        backoff_factor: float,
    ) -> requests.Session:
        """Creates a requests.Session with a retry mechanism."""
        session = requests.Session()
        retry_strategy = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=(500, 502, 503, 504),
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _deep_get(self, d: Dict, keys: List[str]) -> Optional[Any]:
        """Safely retrieves a nested value from a dictionary."""
        for key in keys:
            if isinstance(d, dict):
                d = d.get(key)
            else:
                return None
        return d

    def _get_purpose_from_xml(self, url: str) -> Optional[List[str]]:
        """Fetches and parses a single CMR XML URL to extract the 'purpose' field."""
        try:
            response = self.session.get(url, timeout=20)
            response.raise_for_status()
            xml_dict = xmltodict.parse(response.content)

            id_info = self._deep_get(
                xml_dict,
                ["gmi:MI_Metadata", "gmd:identificationInfo"],
            )
            if not id_info:
                return []

            id_info = id_info if isinstance(id_info, list) else [id_info]
            purposes = []
            for item in id_info:
                purpose = self._deep_get(
                    item,
                    ["gmd:MD_DataIdentification", "gmd:purpose", "gco:CharacterString"],
                )
                if purpose and isinstance(purpose, str):
                    purposes.append(purpose.strip())
            return purposes
        except requests.exceptions.RequestException as e:
            logging.warning(f"Request failed for URL {url}: {e}")
        except xmltodict.expat.ExpatError as e:
            logging.warning(f"XML parsing failed for URL {url}: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred for URL {url}: {e}")
        return None

    def add_purposes_to_dataframe(
        self,
        df: pd.DataFrame,
        url_column: str = "url1",
    ) -> pd.DataFrame:
        """
        Processes URLs in a DataFrame to fetch and add purpose metadata concurrently.

        Args:
            df: The CMR DataFrame.
            url_column: The name of the column containing the XML URLs.

        Returns:
            The DataFrame with a new 'purposes' column.
        """
        urls = df[url_column].dropna()
        results = pd.Series(index=df.index, dtype=object)

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_index = {
                executor.submit(self._get_purpose_from_xml, url): index
                for index, url in urls.items()
            }

            pbar = tqdm(total=len(future_to_index), desc="Fetching CMR Purposes")
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results.at[index] = future.result()
                except Exception as e:
                    logging.error(
                        f"Future for URL at index {index} generated an exception: {e}",
                    )
                pbar.update(1)
            pbar.close()

        df["purposes"] = results
        # Clean up the 'purposes' column
        df["purposes"] = df["purposes"].apply(
            lambda x: " ".join(x) if isinstance(x, list) and x else None,
        )
        df["purposes"] = df["purposes"].apply(
            lambda x: x.strip() if isinstance(x, str) and len(x.strip()) > 0 else None,
        )
        return df


# --- 4. Generic Data Transformation ---


def create_structured_dataset(
    df: pd.DataFrame,
    source_name: str,
    context_map: Dict[str, str],
    metadata_cols: List[str],
) -> pd.DataFrame:
    """
    Transforms a DataFrame into the final structured format.
    Includes a sanitization step to ensure all data is JSON-serializable.

    Args:
        df: The input DataFrame (PDS or CMR).
        source_name: The name of the source ('PDS' or 'CMR').
        context_map: A dictionary mapping column names to labels for the context string.
        metadata_cols: A list of columns to include in the metadata JSON.

    Returns:
        A new DataFrame with columns: 'query', 'context', 'type',
        'synthesized', 'source', and 'metadata'.
    """
    logging.info(f"Transforming data for source: {source_name}")
    transformed_df = pd.DataFrame(index=df.index)

    # 1. Create 'query' and standard columns
    transformed_df["query"] = df["title"]
    transformed_df["synthesized"] = False
    transformed_df["type"] = "title-description"
    transformed_df["source"] = source_name

    # 2. Create 'context' column
    context_parts = []
    for col, label in context_map.items():
        if col in df.columns:
            part = (
                df[col]
                .dropna()
                .astype(str)
                .apply(lambda x: f"{label}: {x}" if x else "")
            )
            context_parts.append(part)
    transformed_df["context"] = pd.concat(context_parts, axis=1).apply(
        lambda x: "\n".join(x.dropna()),
        axis=1,
    )

    # 3. Create 'metadata' JSON column with sanitization
    existing_metadata_cols = [col for col in metadata_cols if col in df.columns]

    # Create a copy to safely modify for JSON serialization
    metadata_df = df[existing_metadata_cols].copy()

    # *** FIX STARTS HERE ***
    # Sanitize the DataFrame to convert non-serializable types
    for col in metadata_df.columns:
        # Check if any cell in the column is a numpy array
        if metadata_df[col].apply(lambda x: isinstance(x, np.ndarray)).any():
            logging.info(
                f"Sanitizing column '{col}' for JSON: converting numpy arrays to lists.",
            )
            # Convert numpy arrays to Python lists; leave other types unchanged
            metadata_df[col] = metadata_df[col].apply(
                lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
            )
    # *** FIX ENDS HERE ***

    metadata_dicts = metadata_df.to_dict(orient="records")

    # This should now work without error
    transformed_df["metadata"] = [json.dumps(d) for d in metadata_dicts]

    # Remove rows where 'context' is None or an empty string
    transformed_df = transformed_df[
        transformed_df["context"].notna() & (transformed_df["context"] != "")
    ]

    final_columns = ["query", "context", "type", "synthesized", "source", "metadata"]
    return transformed_df[final_columns]


def clean_purpose(value):
    """
    Cleans the 'purpose' column based on its type.
    - If it's a list or numpy array, joins the elements. Returns None for an empty one.
    - If it's a string or None, returns it unchanged.
    """
    # CORRECTED: Check if the value is a NumPy array.
    # The type is np.ndarray, not the function np.array.
    if isinstance(value, np.ndarray):
        # If the array is not empty, join its elements into a single string
        if value.size > 0:  # Using .size is a robust way to check for emptiness
            return " ".join(str(item) for item in value)
        # If the array is empty, return None
        else:
            return None
    # Check if the value is a list
    elif isinstance(value, list):
        # If the list is not empty, join its elements into a single string
        if value:
            return " ".join(str(item) for item in value)
        # If the list is empty, return None
        else:
            return None
    # If the value is not a list or array (e.g., a string or None), return it as is
    return value


# --- 5. Main Execution ---


def main():
    """Main function to run the entire data processing pipeline."""
    config = Config()

    # Create output directories if they don't exist
    os.makedirs(config.BASE_DATA_PATH, exist_ok=True)
    os.makedirs(config.OUTPUT_PATH, exist_ok=True)

    # Load and split the data
    cmr_df, pds_df = load_and_split_data(config.SOURCE_CSV_PATH)

    # --- Process PDS Data ---
    logging.info("--- Starting PDS Processing ---")
    pds_subset = pds_df[config.PDS_RAW_COLS]
    pds_aliased = pds_subset.rename(columns=config.COLUMN_ALIAS_MAP)
    pds_pairs = create_structured_dataset(
        df=pds_aliased,
        source_name="PDS",
        context_map=config.PDS_CONTEXT_MAP,
        metadata_cols=config.PDS_METADATA_COLS,
    )
    pds_pairs.to_parquet(config.PDS_OUTPUT_PATH, index=False)
    logging.info(f"✅ PDS data processing complete. Saved to {config.PDS_OUTPUT_PATH}")
    display(pds_pairs.head())

    # --- Process CMR Data (with Caching) ---
    logging.info("\n--- Starting CMR Processing ---")

    if os.path.exists(config.CMR_CACHE_PATH):
        # Cache Hit: Load directly from the cache file
        logging.info(
            f"Cache found at {config.CMR_CACHE_PATH}. Loading pre-processed CMR data.",
        )
        cmr_with_purposes = pd.read_parquet(config.CMR_CACHE_PATH)
    else:
        # Cache Miss: Perform the expensive fetch operation
        logging.info(
            f"Cache not found. Fetching CMR purposes from URLs. This may take a while...",
        )
        cmr_processor = CmrUrlProcessor()
        cmr_with_purposes = cmr_processor.add_purposes_to_dataframe(
            cmr_df,
            url_column="url1",
        )

        # Save the result to the cache for future use
        logging.info(f"Saving fetched CMR data to cache: {config.CMR_CACHE_PATH}")
        cmr_with_purposes.to_parquet(config.CMR_CACHE_PATH, index=False)
    # Clean up the 'purposes' column
    logging.info("Cleaning up 'purposes' column in CMR data.")

    # Clleannig purpose if necessary
    cmr_with_purposes["purposes"] = cmr_with_purposes["purposes"].apply(clean_purpose)

    # Continue with transformation using the (now loaded) cmr_with_purposes DataFrame
    cmr_subset = cmr_with_purposes[
        [col for col in config.CMR_RAW_COLS if col in cmr_with_purposes.columns]
    ]
    cmr_aliased = cmr_subset.rename(columns=config.COLUMN_ALIAS_MAP)

    cmr_pairs = create_structured_dataset(
        df=cmr_aliased,
        source_name="CMR",
        context_map=config.CMR_CONTEXT_MAP,
        metadata_cols=config.CMR_METADATA_COLS,
    )
    cmr_pairs.to_parquet(config.CMR_OUTPUT_PATH, index=False)
    logging.info(f"✅ CMR data processing complete. Saved to {config.CMR_OUTPUT_PATH}")
    display(cmr_pairs.head())


if __name__ == "__main__":
    from IPython.display import display

    main()

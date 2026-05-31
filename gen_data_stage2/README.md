# Stage-2 Data Generation (`gen_data_stage2`)

This module builds the **NASA SDE ST Corpus** — the in-domain synthetic + natural sentence-pair
data used in **Stage 2** of INDUS-SDE-ST training (see the [top-level README](../README.md);
INDUS-SDE / KDD 2026, AI for Sciences Track). It is the home of the generation **schema** and
**system prompt** referenced in the paper appendix.

## Pipeline

1. **Representative-data selection** — [`getting_representative_data/`](getting_representative_data/)
   Embeds candidate SDE documents with a sentence transformer and clusters them (KMeans / t-SNE,
   stratified by NASA SMD division) to pick a diverse, representative subset for generation.
   (`clustering_runner.py`, `1_clustering.ipynb`, `2_division_based.ipynb`, `slurm.sh`.)

2. **LLM content filtering** — [`filter2_llm_based/`](filter2_llm_based/) · **Pydantic AI**
   A Pydantic AI `Agent` with a `ContentQuality` output schema scores each document on information
   density, relevance, accuracy, and structure, gating low-quality / off-domain text before
   generation. (`filter2.py`, `f2.py`; explored in `../flters.ipynb`.)

3. **Schema-enforced pair generation** — [`instructor/`](instructor/) · **Instructor**
   An Instructor-wrapped OpenAI client (`instructor.Mode.JSON`) emits structured
   `DatasetGeneration` objects — grounded query–context pairs plus researcher-oriented search
   terms — directly from each document.
   - `gen_data_instruct.py` — synchronous reference implementation
   - `gen_data_instruct_async.py` — asynchronous (for scale)
   - `gen_data_instruct_async_search_only.py` — search-terms-only variant

   Alternate generators: `pydantic_qa_gen/qa_gen.py` (Pydantic AI `Agent`) and
   `classical_llm/gen_data.py` (plain prompting).

### Frameworks at a glance
| Step | Framework | Why |
|------|-----------|-----|
| Pair / search-term generation | **Instructor** (`Mode.JSON`) | enforces the `DatasetGeneration` schema on the LLM output |
| Content-relevancy filtering & alt. QA gen | **Pydantic AI** (`Agent`) | typed `output_type` agents for scoring / QA |

## Generation schema (Instructor)

Defined in [`instructor/gen_data_instruct.py`](instructor/gen_data_instruct.py):

```python
class QueryContextPair(BaseModel):
    """A generated question (`query`) + the verbatim snippet (`context`) it is answerable from."""
    query: str = Field(
        ...,
        description="A question fully answerable by the 'context'. Mix of factual, "
                    "list/enumeration, inferential, causal, comparative or procedural questions",
    )
    context: str = Field(
        ...,
        description="A verbatim, self-contained passage from the source document containing "
                    "all information needed to answer the 'query'.",
    )

class DatasetGeneration(BaseModel):
    """Top-level response: diverse pairs + curated search terms from one document."""
    question_context: List[QueryContextPair] = Field(
        ...,
        description="Diverse QueryContextPair objects spanning topics, methodologies, and "
                    "findings from across the entire document.",
    )
    search_terms: List[str] = Field(
        ...,
        description="A curated list of high-relevance search terms",
    )
```

## System prompt

`{nquestion}` is the target number of pairs per document (configurable). Verbatim from
[`instructor/gen_data_instruct.py`](instructor/gen_data_instruct.py):

```text
**Your Role:** You are an expert AI specializing in creating datasets for Information Retrieval.

**Your Task:** From the provided text, generate two outputs in the required JSON format:
1.  A list of **approximately {nquestion}** `QueryContextPair` objects.
2.  A list of **approximately {nquestion}** `search_terms`.

**Requirements for Queries:**
    * **Grounded:** Every question MUST be answerable using ONLY the provided `context`. Do not use external knowledge.
    * **Diverse:** Questions must cover a wide range of topics from the text

**Requirements for Context:**
    * **Verbatim:** The `context` MUST be a direct, verbatim quote extracted from the source text.
    * **Sufficient Length:** The context should be a substantial chunk of text (a full paragraph is ideal), not just a single sentence.

**Requirements for Search Terms:**
    * **User Intent:** Think like a researcher. What would they type into Google Scholar or PubMed to find this paper?
    * **Specificity:** Avoid vague, single-word terms.
        * **BAD:** `treatment`, `science`
        * **GOOD:** `mRNA vaccine side effects`, `protein folding accuracy`
    * **Content & Mix:** The list must capture the document's core ideas. It must include a mix of:
        1.  **Technical Phrases:** e.g., 'carbon nanotube synthesis', 'large language model fine-tuning'
        2.  **Named Entities:** e.g., 'CRISPR-Cas9', 'Hubble Space Telescope'
        3.  **Conceptual Queries:** e.g., 'how to improve battery cycle life', 'risks of AI in healthcare'
```
</content>

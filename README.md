# PubMed Clinical Evidence Agent

Give it a clinical note with a diagnosis, and it checks each supporting claim against the literature.
An LLM pulls out the diagnosis and the claims that support it. It then writes a precise PubMed query
for each claim through **tool calling**, fetches the abstracts from NCBI E-utilities, and summarizes
how strongly the evidence supports the claim, with formatted references.

Built during a summer 2025 internship at UCSF (NeuroAI Lab, RISE program).

## Pipeline

```
clinical note ─▶ diagnosis + up to 12 supporting claims (GPT-4o, JSON)
              ─▶ per claim: LLM tool call create_pubmed_search(search_string)  e.g. svPPA → (svPPA OR "semantic variant primary progressive aphasia")[tiab]
              ─▶ esearch → PMIDs → efetch → abstracts   (fallback queries if nothing comes back)
              ─▶ evidence summary + research status (supported / no evidence / no abstracts) + references
```

## Variants

| File | What it changes |
|---|---|
| `simple_pubmed.py` | Core: query generation by tool call, E-utilities search/fetch, summarizing |
| `clinical_diag_exp.py` | Original end-to-end experiment |
| `clinical_diag_exp_enhanced.py` | Diagnosis-first claim extraction, fuzzy search, stricter evidence grading (GPT-4o) |
| `clinical_diag_exp_cost_optimized.py` | Batched claim + query generation on GPT-3.5, fewer LLM calls per claim |
| `clinical_diag_exp_prompts.py` | Prompt-engineering variant |
| `gradio_ui*.py` | Web UI for each variant: paste a note, set the claim and result limits, read the report |

## Run

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...
python gradio_ui_enhanced.py          # or: python clinical_diag_exp_enhanced.py  (runs a synthetic svPPA example)
```

The example note in the code is a synthetic textbook vignette, not a patient record.

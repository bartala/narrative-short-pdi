# narrative-short-pdi

Code for the analyses in the manuscript on screening for probable childbirth-related
post-traumatic stress disorder (CB-PTSD) from birth narratives and Peritraumatic
Distress Inventory (PDI) items.

## Data

The participant data (birth narratives and questionnaire responses) are not included,
to protect participant privacy. Scripts expect a pooled CSV with one row per woman:
`record_id`, `cb_delivery_narrative`, `pdi_q1` to `pdi_q13`, `spcl5_total` (PCL-5
total) and `cohort`. Probable CB-PTSD is defined as a PCL-5 total of 32 or higher.

## Setup

Python 3.10.

```bash
pip install -r requirements.txt
```

Knowledge-graph extraction uses a local LLM through Ollama (`llama3.1:8b`).

## Main analyses

Run from the repository folder, in this order:

```bash
python build_pooled_csv.py --cbex CBEX.csv --covid COVID.csv --out pooled_cohort.csv
python rebuild_dev_kg.py --csv pooled_cohort.csv --old-graph "" --comp-col comp_placeholder --cache pooled_extraction.jsonl --out perignnosis_graph_pooled.pt
python narrative_short_pdi.py --graph perignnosis_graph_pooled.pt --cohort-csv pooled_cohort.csv --out results_narrative_short_pdi/
python pdi_replacement.py --graph perignnosis_graph_pooled.pt --seeds 1 2 3 4 5 --out results_pdi_replacement/
python cross_cohort_validation.py --out results_cross_cohort/
python clinical_utility.py --oof-dir results_narrative_short_pdi/ --model "LR (narrative + six)" --references "LR (PDI-13)" "LR (six)" --band-model "LR (PDI-13)" --cohort-csv pooled_cohort.csv --out results_narrative_short_pdi/clinical_utility/
python calibration_ci.py
python make_fig_calibration_dca.py
python make_fig_roc.py
```

## Supplementary analyses

| Analysis | Scripts |
|---|---|
| PDI items by cohort | `item_auc_by_cohort.py`, `pdi_subset_analysis.py` |
| Narrative embedding models | `embedding_benchmark.py` |
| Graph neural networks and other flexible models | `fair_benchmark.py`, `pca_gnn.py`, `sapbert_entities.py`, `graph_signal_diagnostic.py`, `ground_kg_umls.py`, `popgraph_cs.py` |
| Other graph constructions | `sentence_gnn.py`, `embed_sentences.py`, `word_graph_probe.py`, `word_graph_confirmatory.py`, `word_embed_probe.py`, `word_gnn.py`, `ssl_word_gnn.py`, `symptom_gnn.py`, `centrality_probe.py`, `item_gnn.py` |
| Emotion features | `emotion_features.py`, `emotion_probe.py`, `emotion_confirmatory.py` |
| One model for any PDI short form | `short_form_single_model.py` |

Each script lists its usage at the top of the file.

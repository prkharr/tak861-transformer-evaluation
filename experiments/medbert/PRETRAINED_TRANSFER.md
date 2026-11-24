# Real pretrained biomedical text transfer

Run `archive/notebooks/expanded cohort/10_pretrained_biomedbert_tensor_transfer.ipynb`.
This is **BiomedBERT, not the original structured Med-BERT**. The original authors
state that their weights cannot be shared: https://github.com/ZhiGroup/Med-BERT#sharing-pre-trained-model.
The downloadable Microsoft model is pretrained on biomedical text:
https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext.

## Before running

1. Use an approved Databricks runtime with Spark, the existing Snowflake connector,
   PyTorch >=2.6, NumPy, pandas, scikit-learn, matplotlib, transformers 4.57.6 and
   huggingface_hub >=0.34,<1. Install missing dependencies before starting; restart
   Python after installation. Do not reinstall PyTorch into an active GPU runtime blindly.
2. Fill all seven blank `sf_options` values in the first cell, exactly as in notebook 09.
   Default database/schema are overridden internally to DSVC_TAKEDA_TA_PRIVATE.DS_ML.
3. Keep RUN_ID=R20261030_NEW and RUN_MODE=SUMMARY. Existing 01–04 artifacts are required;
   notebook 05 adds the optional saved LightGBM reference. No new extraction is performed.
4. Allow outbound download from Hugging Face and its artifact CDN. The published weight
   file is about 440 MB. All feature-description inference occurs locally; no patient data
   or feature descriptions are sent to a remote inference service.
5. Run all cells from the beginning. The 3,300-second cooperative budget includes model
   download and tensor loading; blocking network calls and solver fits can exceed it.

The source pins the Microsoft repository revision and verifies SHA-256 for weights,
configuration and vocabulary. No random encoder fallback is permitted. An optional
local mirror requires the same pinned model, tokenizer and configuration files; their
expected hashes are embedded in the code. The notebook records provenance for presentation
to the client without writing into the model directory.

## What it measures

Frozen pretrained text embeddings are computed once for the TRAIN-observed coded
feature names, with underscores replaced by spaces. These names are not a curated
ICD-description dictionary. Inspect the printed examples; poor description quality
can limit usefulness. Count-weighted feature pooling summarizes months 0–2 and 3–11;
all eligible TRAIN-observed groups are retained, without the five-code cap in NB09.
Embeddings are centred on the TRAIN-observed vocabulary before fixed random projection;
mean pairwise cosine is reported to diagnose lack of semantic spread.
New validation-only groups are omitted consistently across all matched arms.

Three logistic heads use identical label-independent bounded TRAIN samples, patient
weights, fixed C, and TRAIN-only scalers: count-only; count plus random embeddings;
count plus pretrained text embeddings. The encoder is frozen, not fine-tuned.
This tests whether a semantic prior improves a regularized classifier, not whether
the original Med-BERT architecture is superior. Convergence warnings are reported.

Every arm scores full frozen validation. Existing saved model predictions are contextual
references trained on larger samples. TEST is not transformed or scored. Validation
has already been used during development; intervals are descriptive 200-draw pilot
patient-cluster intervals, not independent confirmation or production superiority.

Results, checkpoint receipt, numerical heads, embeddings and score vectors are saved
to a new `_BIOMEDBERT_TEXT_V1_<hash>` artifact table. No old experiments are overwritten.
Completed results are reused only after verifying identity and every payload checksum.
Use a driver with sufficient free memory for dense validation features (several GB),
the encoder and the existing tensor archive loader, plus local disk for cached artifacts.
No clinical performance improvement has been measured locally.

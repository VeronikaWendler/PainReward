# %% [markdown]
# ## Analysis of Questionnaire Data and Merging with Behavioural DataFrames

# %% [markdown]
# 1. Importation of Libraries
# 2. Reading in the data_sv_modeling_Coll.csv file which contains various mathematical estimates of subjective value - bayesian code
# 3. Reading in the bahvioural_sv_cleaned.csv file which contains the overall value and absolute value conditions and the estimates of subjective value - frequentist code
# 4. Filter the DataFrame to only keep participants who did not accept money on >= 95% of the trials
# 5. Translate the questionnaire column-headers into English using GoolgeTranslator
# 6. convert the participants_new.tsv that contains SAI,TAI and pcs scales into csv
# 7. Reverse code the selected SAI,TAI columns 
# 8. Sum the column values to get the scores for SAI, TAI and PCS
# 9. Heatmap for SAI, TAI, PCS
# 10. Concatenate the 3 DataFrames (score_data (SAI, TAI, PCS), model_data (fitted sv_pain values), large behavioural DataFrame
# 11. correlating some potential predictors of regression models for the bayesian analysis - heatmap corr.plot
# 
# 12. DataFrames:
# - behavioural_sv_cleaned_final_2.csv
# - data_sv_modeling_Coll.csv
# - behavioural_sv_cleaned_2.csv
# - participants_STAI_PCS_Rev.csv
# - participants_new.csv
# - participants_new.tsv
# - data_sv_modeling_Coll.csv
# - Quest_sv_pain_Corr.png
# 

# %%
# importing libraries
import pandas as pd
import os
from pathlib import Path
from os.path import join as opj


basepath = str(os.getenv("basepath", Path(__file__).parent.parent.parent))


# =========================
# Load + clean headers
# =========================
df = pd.read_csv(opj(basepath, 'participants.tsv'), sep="\t")

df.columns = (
    df.columns.astype(str)
    .str.replace("\ufeff", "", regex=False)
    .str.replace("\u00a0", " ", regex=False)
    .str.replace("’", "'", regex=False)
    .str.strip()
)

# # Drop qsocio columns
# df = df.loc[:, ~df.columns.str.contains("qsocio", case=False, na=False)]

# =========================
# Define columns (exactly as your CSV)
# =========================
TAI_ITEMS = [c for c in df.columns if c.startswith("qiastay2_")]

SAI_ITEMS = [c for c in df.columns if c.startswith("qiastay1_")]

PCS_ITEMS = [c for c in df.columns if c.startswith("qpcs_") and "date" not in c.lower() and "intitulé" not in c.lower()]

assert len(PCS_ITEMS) == 13, f"Expected 13 PCS items, found {len(PCS_ITEMS)}. Check your column names."

assert len(TAI_ITEMS) == 20, f"Expected 20 TAI items, found {len(TAI_ITEMS)}. Check your column names."

assert len(SAI_ITEMS) == 20, f"Expected 20 SAI items, found {len(SAI_ITEMS)}. Check your column names."

TAI_REVERSE = [
    "qiastay2_Je me sens bien",
    "qiastay2_Je me sens content(e) de moi-même",
    "qiastay2_Je me sens reposé(e)",
    "qiastay2_Je suis d'un grand calme",
    "qiastay2_Je suis heureux(se)",
    "qiastay2_Je me sens en sécurité",
    "qiastay2_Prendre des décisions m'est facile",
    "qiastay2_Je suis satisfait(e)",
    "qiastay2_Je suis une personne qui a les nerfs solides",
]

SAI_REVERSE = [
    "qiastay1_Je me sens calme",
    "qiastay1_Je me sens en sécurité",
    "qiastay1_ Je me sens tranquille",
    "qiastay1_ Je me sens comblé(e)",
    "qiastay1_Je me sens à l'aise",
    "qiastay1_Je me sens sûr(e) de moi",
    "qiastay1_Je suis détendu(e)",
    "qiastay1_Je me sens satisfait(e)",
    "qiastay1_Je sens que j'ai les nerfs solides",
    "qiastay1_Je me sens bien",
]

# Safety: keep only cols that exist
TAI_ITEMS   = [c for c in TAI_ITEMS if c in df.columns]
SAI_ITEMS   = [c for c in SAI_ITEMS if c in df.columns]
PCS_ITEMS   = [c for c in PCS_ITEMS if c in df.columns]
TAI_REVERSE_col = [c for c in TAI_REVERSE if c in df.columns]
SAI_REVERSE_col = [c for c in SAI_REVERSE if c in df.columns]

assert len(TAI_REVERSE_col) == len(TAI_REVERSE)
assert len(SAI_REVERSE_col) == len(SAI_REVERSE)

df[TAI_ITEMS] = df[TAI_ITEMS].apply(pd.to_numeric, errors="coerce")
df[SAI_ITEMS] = df[SAI_ITEMS].apply(pd.to_numeric, errors="coerce")
df[PCS_ITEMS] = df[PCS_ITEMS].apply(pd.to_numeric, errors="coerce")
# =========================
# Reverse-score (1..4)
# =========================
df[TAI_REVERSE] = 5 - df[TAI_REVERSE]
df[SAI_REVERSE] = 5 - df[SAI_REVERSE]

# =========================
# FINAL SCORES
# =========================
# 20..80 range (this is what you want now)
df["STA_TAI_Score"] = df[TAI_ITEMS].sum(axis=1)
df["STA_SAI_Score"] = df[SAI_ITEMS].sum(axis=1)

# Optional: keep the old shifted versions too (0..60) for reference
df["STA_TAI_minus20"] = df["STA_TAI_Score"] - 20
df["STA_SAI_minus20"] = df["STA_SAI_Score"] - 20

# PCS: sum (unchanged)
if PCS_ITEMS:
    df["PCS_Score"] = df[PCS_ITEMS].sum(axis=1)


df['PCS_Score'] = df['PCS_Score']
df['STA_SAI_Score'] = df['STA_SAI_Score']
df['STA_TAI_Score'] = df['STA_TAI_Score']

# Overwrite
df.to_csv(opj(basepath, "participants.tsv"), index=False, sep="\t")



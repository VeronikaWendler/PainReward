import pandas as pd

df = pd.read_csv(r"D:/GitHub/PainReward_ULaval/Hddm_Docker_August_24/data_sets/behavioural_sv_cleaned_final_3.csv")
df['subj_idx'] = df['subj_idx'].str.extract(r'(\d+)').astype(int)

df.to_csv(r"D:/GitHub/PainReward_ULaval/Hddm_Docker_August_24/data_sets/behavioural_sv_cleaned_final_3.csv", index=False)

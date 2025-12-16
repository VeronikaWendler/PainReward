  
# --- MACHINE LEARNING ON FUNCTIONAL LOCALIZER DATA ---
import numpy np 
import pandas as pd 


bin_width_ms = 10

sfreq = epochs_func_loc.info[‘sfreq’]
samples_per_bin = int((bin_width_ms / 1000) * sfreq)
print(f”\nFor {sub}, using bin width of {bin_width_ms}ms ({samples_per_bin} samples).“)

epochs_for_training = epochs_func_loc.copy().pick(‘eeg’)
print(f”Channels selected for training: {len(epochs_for_training.ch_names)}“)

X = epochs_for_training.get_data()
y = epochs_for_training.metadata[‘image_file’]

model_ch_names = epochs_for_training.ch_names
n_times = X.shape[2]
pipeline = make_pipeline(Vectorizer(), StandardScaler(), LogisticRegression(solver=‘liblinear’, random_state=42, max_iter=1000))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
bin_centers_time, all_binned_scores, std_binned_scores = [], [], []
for i in range(0, n_times - samples_per_bin + 1, samples_per_bin):
    X_bin = X[:, :, i:i + samples_per_bin]
    bin_center_time_sec = epochs_for_training.times[i + samples_per_bin // 2]
    bin_centers_time.append(bin_center_time_sec)
    scores_bin = cross_val_score(pipeline, X_bin, y, cv=cv, scoring=‘accuracy’, n_jobs=-1)
    all_binned_scores.append(scores_bin.mean())
    std_binned_scores.append(scores_bin.std())
all_binned_scores = np.array(all_binned_scores)
std_binned_scores = np.array(std_binned_scores)
chance_level = 1 / len(np.unique(y))
plt.figure(figsize=(12, 6))
plt.plot(bin_centers_time, all_binned_scores, label=‘Mean Binned Decoding Accuracy’)
plt.axhline(chance_level, color=‘red’, linestyle=‘--‘, label=f’Chance Accuracy ({chance_level:.2f})’)
plt.axvline(0, color=‘black’, linestyle=‘-.’, label=‘Stimulus Onset (t=0)’)
plt.fill_between(bin_centers_time, all_binned_scores - std_binned_scores, all_binned_scores + std_binned_scores,
                 alpha=0.2, color=‘blue’, label=‘±1 Standard Deviation’)
plt.title(f’{sub} - Binned ({bin_width_ms}ms) Time-Resolved Decoding Accuracy’)
plt.xlabel(‘Time (s)’); plt.ylabel(‘Classifier Accuracy’); plt.legend(loc=‘upper left’); plt.grid(True, linestyle=‘:’)
plt.savefig(os.path.join(sub_folder, f’{sub}_binned_time_resolved_decoding_accuracy.png’), dpi=300)
plt.show()
peak_bin_idx = np.argmax(all_binned_scores)
peak_time_sec = bin_centers_time[peak_bin_idx]
print(f”Peak decoding accuracy of {all_binned_scores.max():.3f} found at {peak_time_sec:.3f}s.“)
start_sample_peak = peak_bin_idx * samples_per_bin
X_best_bin = X[:, :, start_sample_peak:start_sample_peak + samples_per_bin]
import pickle
import pandas as pd
import numpy as np

# 23-channel alphabet (match training)
amino_acids = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard
extra_tokens = ["X", "U", "O"]        # add 3 extras
full_vocab = list(amino_acids) + extra_tokens

def one_hot(seq):
    encoding = np.zeros((len(seq), len(full_vocab)), dtype=np.float32)
    for i, aa in enumerate(seq):
        if aa in full_vocab:
            encoding[i, full_vocab.index(aa)] = 1
        else:
            # unknown → map to X channel
            encoding[i, full_vocab.index("X")] = 1
    return encoding

df = pd.read_csv("data/suc/test_data.csv")
sequences = df["sequence"].tolist()

features = {seq: one_hot(seq) for seq in sequences}

with open("data/suc/onehot_features.pkl", "wb") as f:
    pickle.dump(features, f)

print("One-hot features saved (23 channels).")

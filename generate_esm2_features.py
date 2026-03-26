import torch
import esm
import pickle
import pandas as pd
from tqdm import tqdm

# Load your peptides from CSV
df = pd.read_csv("data/suc/test_data.csv")

sequences = df["sequence"].tolist()

# Load pretrained ESM-2 model
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
model.eval()

batch_converter = alphabet.get_batch_converter()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

features = {}

for seq in tqdm(sequences):
    data = [("protein", seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6])
        token_representations = results["representations"][6]

    # Remove CLS and EOS tokens
    embedding = token_representations[0, 1:len(seq)+1].cpu().numpy()
    features[seq] = embedding

# Save to pickle
with open("data/suc/esm2_features.pkl", "wb") as f:
    pickle.dump(features, f)

print("ESM2 features saved.")

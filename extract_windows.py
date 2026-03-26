from Bio import SeqIO
import pandas as pd

WINDOW = 33
HALF = 16

input_fasta = "data/suc/mouse_myosin.fasta"
output_csv = "data/suc/test_data.csv"

rows = []

for record in SeqIO.parse(input_fasta, "fasta"):
    seq = str(record.seq)

    for i, aa in enumerate(seq):
        if aa == "K":
            start = i - HALF
            end = i + HALF + 1

            if start >= 0 and end <= len(seq):
                peptide = seq[start:end]

                rows.append({
                    "protein_id": record.id,
                    "lysine_position_1based": i + 1,
                    "sequence": peptide,
                    "label": 0
                })

df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False)

print(f"Saved {len(df)} peptides to {output_csv}")

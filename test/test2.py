import pandas as pd

df = pd.read_csv('data/AGILE_smiles_with_value_group.csv')
# 在df中找到arg.mol1对应的combined_mol_SMILES
label = 'A1B1C1'
smiles = df[df['label'] == label]['combined_mol_SMILES'].values[0]

print(smiles)
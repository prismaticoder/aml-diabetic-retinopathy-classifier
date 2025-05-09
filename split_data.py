import pandas as pd
from sklearn.model_selection import train_test_split

# Load your full training labels CSV
df = pd.read_csv('dataset/trainLabels.csv')

# Split into train (80%), validation (10%), and test (10%) stratified by the label
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['level'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['level'])

# Save new CSVs
train_df.to_csv('dataset/train.csv', index=False)
val_df.to_csv('dataset/val.csv', index=False)
test_df.to_csv('dataset/test.csv', index=False)

print("✅ train.csv, val.csv, and test.csv successfully created.")

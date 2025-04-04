import pandas as pd
from sklearn.model_selection import train_test_split

# Load your full training labels CSV
df = pd.read_csv('dataset/trainLabels.csv')

# Split into train (80%) and val (20%) stratified by the label
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['level']
)

# Save new CSVs
train_df.to_csv('dataset/train.csv', index=False)
val_df.to_csv('dataset/val.csv', index=False)

print("✅ train.csv and val.csv successfully created.")

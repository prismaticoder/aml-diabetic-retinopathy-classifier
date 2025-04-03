import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_dataset(input_csv='dataset/trainLabels.csv', output_dir='labels', train_size=0.7, val_size=0.15):
    """
    Split dataset into train, validation and test sets.
    
    Args:
        input_csv (str): Path to input CSV file
        output_dir (str): Directory to save split CSV files
        train_size (float): Proportion for training set
        val_size (float): Proportion for validation set (test_size will be the remainder)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # First split: separate train from rest
    train_df, temp_df = train_test_split(
        df, 
        train_size=train_size,
        random_state=42,
        stratify=df['level'] if 'level' in df.columns else None
    )
    
    # Second split: divide remaining data into val and test
    # Calculate relative validation size from remaining data
    remaining_val_size = val_size / (1 - train_size)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=remaining_val_size,
        random_state=42,
        stratify=temp_df['level'] if 'level' in temp_df.columns else None
    )
    
    # Save the splits to CSV files
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    # Print split sizes
    print(f"Total samples: {len(df)}")
    print(f"Train samples: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation samples: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test samples: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

if __name__ == "__main__":
    split_dataset()

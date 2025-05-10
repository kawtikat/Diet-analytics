import pandas as pd
from sklearn.model_selection import train_test_split

def split_and_save_dataset(df, train_size=0.8, random_state=42):
    """
    Randomly scramble the dataset, split into train/test sets, and save to files
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    train_size (float): Proportion of data to use for training (default 0.8)
    random_state (int): Random seed for reproducibility
    """
    
    # First, shuffle the entire dataset
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Split into training and testing sets
    train_df, test_df = train_test_split(
        df_shuffled,
        train_size=train_size,
        random_state=random_state
    )
    
    # Reset indices for both datasets
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    # Save to CSV files
    train_df.to_csv('food_train_data.csv', index=False)
    test_df.to_csv('food_test_data.csv', index=False)
    
    # Print information about the split
    print(f"Original dataset shape: {df.shape}")
    print(f"Training dataset shape: {train_df.shape}")
    print(f"Testing dataset shape: {test_df.shape}")
    
    # Print the first few rows of each dataset
    print("\nFirst few rows of training dataset:")
    print(train_df.head())
    print("\nFirst few rows of testing dataset:")
    print(test_df.head())
    
    return train_df, test_df

# Example usage:
# df = pd.read_csv('your_food_data.csv')
# train_data, test_data = split_and_save_dataset(df)
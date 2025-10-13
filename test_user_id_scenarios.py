import pandas as pd
import numpy as np

# Load the actual client_ratings.xlsx file
print("Loading client_ratings.xlsx...")
df_client_ratings = pd.read_excel('data/client_ratings.xlsx')

print("=== Current client_ratings.xlsx structure ===")
print(f"Shape: {df_client_ratings.shape}")
print(f"Columns: {df_client_ratings.columns.tolist()[:10]}...")  # Show first 10 columns
print("\nFirst few rows:")
print(df_client_ratings[['Timestamp', 'Enter name ', 'Mobile no', 'Age']].head())
print()

# Check if user_id column already exists
if 'user_id' not in df_client_ratings.columns:
    print("Adding user_id column to client_ratings.xlsx...")
    
    # Add user_id column starting from 1
    df_client_ratings.insert(0, 'user_id', range(1, len(df_client_ratings) + 1))
    
    # Save the updated file
    df_client_ratings.to_excel('data/client_ratings.xlsx', index=False)
    
    print("✅ Successfully added user_id column!")
    print(f"user_id range: 1 to {len(df_client_ratings)}")
else:
    print("❌ user_id column already exists!")

print("\n=== Updated client_ratings.xlsx structure ===")
print(f"Shape: {df_client_ratings.shape}")
print(f"Columns: {df_client_ratings.columns.tolist()[:10]}...")  # Show first 10 columns
print("\nFirst few rows with user_id:")
print(df_client_ratings[['user_id', 'Timestamp', 'Enter name ', 'Mobile no', 'Age']].head())

# Example scenarios for testing
print("\n" + "="*50)
print("EXAMPLE SCENARIOS FOR TESTING")
print("="*50)

# Example 1: Excel without user_id column (for demonstration)
data_without_user_id = {
    'Enter name ': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'The Matrix': [5, np.nan, 4, 3],
    'Inception': [np.nan, 4, 5, np.nan],
    'Interstellar': [4, 3, np.nan, 5],
    'Avatar': [np.nan, 5, 3, 4]
}

df_no_user_id = pd.DataFrame(data_without_user_id)
print("=== Example: Excel WITHOUT user_id column ===")
print(df_no_user_id)
print()

# Example 2: Excel WITH user_id column (production setup)
data_with_user_id = {
    'user_id': [101, 102, 103, 104],
    'Enter name ': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'The Matrix': [5, np.nan, 4, 3],
    'Inception': [np.nan, 4, 5, np.nan],
    'Interstellar': [4, 3, np.nan, 5],
    'Avatar': [np.nan, 5, 3, 4]
}

df_with_user_id = pd.DataFrame(data_with_user_id)
print("=== Example: Excel WITH user_id column ===")
print(df_with_user_id)
print()

# Save example files for reference
df_no_user_id.to_excel('example_no_user_id.xlsx', index=False)
df_with_user_id.to_excel('example_with_user_id.xlsx', index=False)

print("Created example files:")
print("- example_no_user_id.xlsx (system will create user_id: 1,2,3,4)")
print("- example_with_user_id.xlsx (system will use user_id: 101,102,103,104)")
print("\n✅ Script completed successfully!")
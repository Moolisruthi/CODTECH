import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Handle missing values and encode categorical variables."""
    
    # Handle missing numerical values
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if numeric_cols.empty:
        raise ValueError("No numeric columns found in the dataset. Please check the input data.")
    
    imputer = SimpleImputer(strategy="mean")
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=["object"]).columns
    if categorical_cols.any():
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # Updated argument
        encoded_data = encoder.fit_transform(df[categorical_cols])
        df_encoded = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
        df = df.drop(columns=categorical_cols).reset_index(drop=True)
        df = pd.concat([df, df_encoded], axis=1)
    
    return df

def transform_data(df):
    """Scale numerical features."""
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled

def save_data(df, output_path):
    """Save the processed data to a CSV file."""
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

def main():
    input_file = "raw_data.csv"  # Ensure this file exists
    output_file = "processed_data.csv"
    
    try:
        # Extract
        df = load_data(input_file)
        
        # Preprocess
        df_preprocessed = preprocess_data(df)
        
        # Transform
        df_transformed = transform_data(df_preprocessed)
        
        # Load
        save_data(df_transformed, output_file)
    except Exception as e:
        print(f"ETL process failed: {e}")

if __name__ == "__main__":
    main()

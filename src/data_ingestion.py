
import pandas as pd

# Data ingestion

def data_ingest():
    """Ingesting data for Raptor indexing"""

    file_paths=[
        "C:\All_projects\PATH_Opensource\chunk1.csv"

    ]

    docs = []

    for file_path in file_paths:
        df = pd.read_csv(file_path)

        if 'Chunk' in df.columns:
            docs.extend(df['Chunk'].tolist())
        else:
            print(f"Column 'content' not found in {file_path}")
            
    return docs

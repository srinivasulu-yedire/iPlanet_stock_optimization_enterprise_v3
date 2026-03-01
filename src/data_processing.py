import pandas as pd
from utils.logger import log

def process_data(df, store_val=None, product_val=None):
    log("=== DATA PROCESSING STARTED ===")
    
    # 1. Strip spaces from all string values in the dataframe
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()

    # 2. Rename columns to internal names
    rename_map = {
        "BillDate": "date",
        "ProductName": "product",
        "Quantity": "sales",
        "BusinessSegmentName": "store"
    }
    df = df.rename(columns=rename_map)

    # 3. Handle Dates
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['date']).sort_values("date")

    # 4. Filter
    if store_val and store_val != "ALL":
        df = df[df["store"] == str(store_val)]

    if product_val and product_val != "ALL":
        df = df[df["product"] == str(product_val)]

    # 5. Daily Aggregation
    df = df.groupby(['date', 'product']).agg({'sales': 'sum'}).reset_index()
    
    # Time features
    df["day"], df["month"], df["year"] = df["date"].dt.day, df["date"].dt.month, df["date"].dt.year

    log(f"Processing complete: {len(df)} rows.")
    return df
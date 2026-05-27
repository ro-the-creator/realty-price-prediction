import pandas as pd
import numpy as np
from rapidfuzz import fuzz


def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={'zip_code': 'postal_code'})


def filter_new_york(df: pd.DataFrame, threshold: int = 80) -> pd.DataFrame:
    target = 'New York'
    df['state_similarity'] = df['state'].apply(lambda x: fuzz.ratio(x, target))
    df_ny = df[df['state_similarity'] > threshold].copy()
    df_ny.drop(columns='state_similarity', inplace=True)
    return df_ny


def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    critical_cols = ['bed', 'bath', 'price', 'house_size', 'acre_lot']
    return df.dropna(subset=critical_cols)


def remove_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    for col in ['bed', 'price', 'acre_lot', 'house_size']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df


def cap_bathrooms(df: pd.DataFrame, max_bath: int = 10) -> pd.DataFrame:
    return df[df['bath'] <= max_bath]


def convert_to_int(df: pd.DataFrame) -> pd.DataFrame:
    int_cols = ['price', 'bed', 'bath', 'house_size']
    df = df.copy()
    for col in int_cols:
        df[col] = df[col].astype(int)
    return df


def target_encode_city(df: pd.DataFrame, alpha: int = 10) -> pd.DataFrame:
    df = df.copy()
    overall_mean_price = df['price'].mean()
    city_stats = df.groupby('city')['price'].agg(['mean', 'count'])
    city_stats['city_te'] = (
        city_stats['mean'] * city_stats['count'] + overall_mean_price * alpha
    ) / (city_stats['count'] + alpha)
    df['city_te'] = round(df['city'].map(city_stats['city_te']), 2)
    return df


def run_pipeline(input_path: str, output_path: str) -> pd.DataFrame:
    df = load_data(input_path)
    df = rename_columns(df)
    df = filter_new_york(df)
    df = drop_missing(df)
    df = remove_outliers_iqr(df)
    df = cap_bathrooms(df)
    df = convert_to_int(df)
    df = target_encode_city(df)
    df.to_csv(output_path, index=False)
    print(f"Pipeline complete. Output saved to: {output_path}")
    print(f"Final shape: {df.shape}")
    return df


if __name__ == '__main__':
    INPUT_PATH = 'pipeline/data/input/realtor-data.csv'
    OUTPUT_PATH = 'pipeline/data/output/cleaned_ny_listings.csv'
    run_pipeline(INPUT_PATH, OUTPUT_PATH)

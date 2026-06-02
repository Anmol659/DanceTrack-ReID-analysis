import pandas as pd

df = pd.read_parquet(
    r"C:\Users\anmol\Desktop\dancetrack\data\metadata\annotations.parquet"
)

print(df.columns.tolist())

print(df.head())
import pandas as pd
try:
    df = pd.read_json('data.json', orient='index')
    print("Columns:", df.columns)
    print("Head:", df.head())
    print("'user_uuid' in columns:", 'user_uuid' in df.columns)
except Exception as e:
    print("Error:", e)


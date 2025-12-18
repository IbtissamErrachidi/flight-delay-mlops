import os
import pandas as pd

def load_and_sample_data(
    dataset_dir,
    max_rows_per_year=100_000,
    chunksize=500_000,
    usecols=None
):
    yearly_data = {}

    for file in os.listdir(dataset_dir):
        if not file.endswith(".csv"):
            continue

        print("Lecture :", file)
        path = os.path.join(dataset_dir, file)

        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
            chunk = chunk.dropna(subset=['ARR_DELAY'])
            chunk["year"] = pd.to_datetime(chunk["FL_DATE"]).dt.year
            chunk["month"] = pd.to_datetime(chunk["FL_DATE"]).dt.month

            for year, group_year in chunk.groupby("year"):
                if year not in yearly_data:
                    yearly_data[year] = {m: [] for m in range(1, 13)}

                max_per_month = max_rows_per_year // 12

                for month, group_month in group_year.groupby("month"):
                    already = sum(len(df) for df in yearly_data[year][month])
                    if already >= max_per_month:
                        continue

                    needed = max_per_month - already
                    take = min(len(group_month), needed)
                    sampled = group_month.sample(take, random_state=42)
                    yearly_data[year][month].append(sampled)

    final_list = []
    for year in yearly_data:
        for month in yearly_data[year]:
            if yearly_data[year][month]:
                final_list.append(
                    pd.concat(yearly_data[year][month], ignore_index=True)
                )

    df_final = pd.concat(final_list, ignore_index=True)
    print("Shape finale :", df_final.shape)

    return df_final

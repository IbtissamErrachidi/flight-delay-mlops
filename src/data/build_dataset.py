def build_dataset(df):
    rename_dict = {
        "FL_DATE": "flight_date",
        "OP_CARRIER": "airline",
        "OP_CARRIER_FL_NUM": "flight_number",
        "ORIGIN": "origin_airport",
        "DEST": "dest_airport",
        "CRS_DEP_TIME": "scheduled_dep_time",
        "DEP_DELAY": "dep_delay",
        "TAXI_OUT": "taxi_out",
        "WHEELS_OFF": "wheels_off",
        "WHEELS_ON": "wheels_on",
        "TAXI_IN": "taxi_in",
        "CRS_ARR_TIME": "scheduled_arr_time",
        "CRS_ELAPSED_TIME": "scheduled_elapsed_time",
        "DISTANCE": "distance",
        "CANCELLED": "cancelled",
        "DIVERTED": "diverted",
        "year": "year",
        "month": "month",
        "ARR_DELAY": "arr_delay"
    }

    df = df.rename(columns=rename_dict)
    df = df.dropna(subset=['dep_delay'])

    y = df.pop('arr_delay')
    X = df

    return X, y
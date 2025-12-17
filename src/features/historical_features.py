import pandas as pd


def add_historical_features(X_train, y_train, X):
    """
    Ajoute les features historiques à X (peut être train ou test)
    sans fuite de données.

    Features ajoutées :
    - carrier_arr_delay_mean : retard moyen par compagnie
    - origin_arr_delay_mean  : retard moyen par aéroport d'origine
    - arr_hour_delay_mean    : retard moyen par heure d'arrivée
    """

    X = X.copy()

    # S'assurer que y_train est un DataFrame
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_frame(name="arr_delay")
    else:
        y_train = y_train.copy()
        y_train.columns = ["arr_delay"]

    # Retard moyen par compagnie
    carrier_mean = y_train.join(X_train[['airline']], how='right') \
                          .groupby('airline')['arr_delay'].mean()
    X['carrier_arr_delay_mean'] = X['airline'].map(carrier_mean).fillna(carrier_mean.mean())

    # Retard moyen par aéroport d'origine
    origin_mean = y_train.join(X_train[['origin_airport']], how='right') \
                         .groupby('origin_airport')['arr_delay'].mean()
    X['origin_arr_delay_mean'] = X['origin_airport'].map(origin_mean).fillna(origin_mean.mean())

    # Retard moyen par heure d'arrivée
    arr_hour_mean = y_train.join(X_train[['arr_hour']], how='right') \
                           .groupby('arr_hour')['arr_delay'].mean()
    X['arr_hour_delay_mean'] = X['arr_hour'].map(arr_hour_mean).fillna(arr_hour_mean.mean())

    return X

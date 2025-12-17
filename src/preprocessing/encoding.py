from sklearn.preprocessing import OrdinalEncoder
import joblib

def encode_categorical(X_train, X_test, cat_cols):
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = encoder.transform(X_test[cat_cols])
    joblib.dump(encoder, "models/encoder.pkl")

    return X_train, X_test, encoder

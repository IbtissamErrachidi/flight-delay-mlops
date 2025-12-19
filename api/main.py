from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict

# Créer l'application FastAPI
app = FastAPI(title="Flight Delay Prediction API")

# Définir le Pydantic model
class FlightInput(BaseModel):
    flight_date: str
    airline: int
    flight_number: int
    origin_airport: int
    dest_airport: int
    scheduled_dep_time: float
    dep_delay: float
    taxi_out: float
    wheels_off: float
    scheduled_arr_time: float
    scheduled_elapsed_time: float
    distance: float
    year: int
    month: int

# Route principale de test
@app.get("/")
def read_root():
    return {"message": "API is running!"}

# Route de prédiction
@app.post("/predict")
def predict_delay(flight: FlightInput):
    pred = predict(flight)
    return {"predicted_arrival_delay": float(pred)}



if __name__ == "__main__":
    
    example_row = {
    "flight_date": "2025-12-19",   # date du vol
    "airline": 19,                 # code de la compagnie
    "flight_number": 3202,         # numéro du vol
    "origin_airport": 305,         # code aéroport de départ
    "dest_airport": 56,            # code aéroport d'arrivée
    "scheduled_dep_time": 1420.0,  # heure prévue de départ en format HHMM (14:20)
    "dep_delay": 5.0,              # retard au départ en minutes
    "taxi_out": 18.0,              # temps entre gate et décollage en minutes
    "wheels_off": 1432.0,          # heure réelle du décollage HHMM
    "scheduled_arr_time": 1600.0,  # heure prévue d'arrivée HHMM
    "scheduled_elapsed_time": 100.0, # durée prévue du vol en minutes
    "distance": 450.0,             # distance entre aéroports
    "year": 2025,
    "month": 12
}

    # Créer un objet FlightInput
    flight_input = FlightInput(**example_row)

    # Prédire
    predicted_delay = predict(flight_input)

    print("Données d'entrée :")
    print(flight_input)
    print("\nPrédiction du retard d'arrivée :", predicted_delay)

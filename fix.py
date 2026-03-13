# fixing binary model issue
import joblib
import cloudpickle

# Load existing models (trained on your mac)
rf = joblib.load("models/random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")
le = joblib.load("models/label_encoder.pkl")

# Save them with cloudpickle
with open("models/random_forest_cp.pkl", "wb") as f:
    cloudpickle.dump(rf, f)

with open("models/scaler_cp.pkl", "wb") as f:
    cloudpickle.dump(scaler, f)

with open("models/label_encoder_cp.pkl", "wb") as f:
    cloudpickle.dump(le, f)

print("Models re-saved with cloudpickle!")
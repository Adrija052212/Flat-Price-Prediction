from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Get the folder where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model and encoder
model = joblib.load(
    os.path.join(BASE_DIR, "model", "flat_price_model.pkl")
)

encoder = joblib.load(
    os.path.join(BASE_DIR, "model", "encoder.pkl")
)


@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        # Get values from the form
        flat_id = request.form["flat_id"]
        area = float(request.form["area"])
        facing = request.form["facing"]
        floor = float(request.form["floor"])
        parking = float(request.form["parking"])
        bedrooms = float(request.form["bedrooms"])

        # Encode facing
        encoded = encoder.transform([[facing]])

        # Numerical features
        numerical_data = np.array([
            [area, floor, parking, bedrooms]
        ])

        # Combine numerical + encoded features
        input_data = np.hstack((numerical_data, encoded))

        # Make prediction
        prediction = model.predict(input_data)

        predicted_price = round(prediction[0], 2)

        # Send result back to HTML
        return render_template(
            "index.html",
            prediction=predicted_price,
            flat_id=flat_id,
            area=area,
            facing=facing,
            floor=floor,
            parking=parking,
            bedrooms=bedrooms
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)

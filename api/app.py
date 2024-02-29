from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def render():
    print("HIEP")
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    scaler = MinMaxScaler()
    train_data = pd.read_csv("train_data.csv").values[:, 1:]
    print(train_data)
    scaler.fit(train_data)
    age = int(request.form["age"])
    height = int(request.form["height"])
    weight = int(request.form["weight"])
    avg_glucozo = float(request.form["avg_glucozo"])
    gender = request.form["gender"]
    heart_deseart = request.form["heart_deseart"]
    hypertension = request.form["hypertension"]
    smoke = request.form["smoke"]
    input = [0] * 11
    if gender == "Nam":
        input[-1] = 1
    else:
        input[-2] = 1
    if heart_deseart == "Có":
        input[2] = 1
    if hypertension == "Có":
        input[1] = 1
    if smoke == "Không biết":
        input[5] = 1
    elif smoke == "Đã từng":
        input[6] = 1
    elif smoke == "Chưa bao giờ":
        input[7] = 1
    else:
        input[8] = 1
    input[0] = age
    input[3] = avg_glucozo
    input[4] = weight / (height / 100**2)
    input_sc = scaler.transform([input])
    result = model.predict(input_sc)[0]
    if result == 1:
        return render_template(
            "index.html", pred="Bạn được chẩn đoán là có khả năng mắc bệnh"
        )
    else:
        return render_template("index.html", pred="Bạn không bị bệnh")


if __name__ == "__main__":
    app.run()

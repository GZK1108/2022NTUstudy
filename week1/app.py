from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        rates = float(request.form.get("rates"))
        model = joblib.load("regression.j1")
        r = model.predict([[rates]])
        return render_template("indextest.html", result=r)
    else:
        return render_template("indextest.html", result="WAITING")

if __name__ == "__main__":
    app.run()
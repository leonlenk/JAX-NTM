# run with: flask --app GUI.app run --debug

from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("pages/index.html")


@app.route("/models")
def configure_models():
    return render_template("pages/models.html")


@app.route("/submit", methods=["POST"])
def submit():
    data = request.form["learning_rate"]
    return f"Result: {float(data)}"


if __name__ == "__main__":
    app.run(debug=True)

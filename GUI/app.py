# run with: flask --app GUI.app run --debug

from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("pages/index.html")


@app.route("/models")
def get_models():
    return render_template("pages/models.html")


if __name__ == "__main__":
    app.run(debug=True)

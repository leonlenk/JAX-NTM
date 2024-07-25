# run with: flask --app GUI.app run --debug

from flask import Flask, render_template, request

from GUI import config_options, parse_requests

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("pages/index.html")


@app.route("/training")
def configure_models():
    optimizers_options = config_options.OPTIMIZERS.keys()
    memory_options = config_options.MEMORY_MODELS.keys()
    read_controller_options = config_options.READ_CONTROLLERS.keys()
    write_controller_options = config_options.WRITE_CONTROLLERS.keys()
    datasets = config_options.DATA_SET.keys()
    curriculums = config_options.CURRICULUM.keys()

    return render_template(
        "pages/training.html",
        optimizers_options=optimizers_options,
        memory_options=memory_options,
        read_controller_options=read_controller_options,
        write_controller_options=write_controller_options,
        datasets=datasets,
        curriculums=curriculums,
    )


@app.route("/submit", methods=["POST"])
def submit():
    # curriculum_config = request.form["curriculum"]

    parse_requests.process_training_request(request.form)

    return "hi"


if __name__ == "__main__":
    app.run(debug=True)

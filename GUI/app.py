# run with: flask --app GUI.app run --debug

from flask import Flask, render_template, request

from GUI import config_options
from Training.Backbones.NTM_graves2014 import LSTMConfig, LSTMModel

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
    lstm_config = LSTMConfig(
        learning_rate=float(request.form["learning_rate"]),
        optimizer=config_options.OPTIMIZERS[request.form["optimizer"]],
        memory_M=int(request.form["memory_m"]),
        memory_N=int(request.form["memory_n"]),
        memory_class=config_options.MEMORY_MODELS[request.form["memory_model"]],
        backbone_class=LSTMModel,
        read_head_class=config_options.READ_CONTROLLERS[
            request.form["read_controller"]
        ],
        write_head_class=config_options.WRITE_CONTROLLERS[
            request.form["write_controller"]
        ],
        num_layers=int(request.form["layers"]),
        input_features=1,
    )

    # curriculum_config = request.form["curriculum"]

    return f"Result: {lstm_config}"


if __name__ == "__main__":
    app.run(debug=True)

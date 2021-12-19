import os
import subprocess

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

from calc_zscore import neuro

UPLOAD_FOLDER = os.getcwd() + "/out"
ALLOWED_EXTENSIONS = set(["m00"])

app = Flask(__name__)

if __name__ == "__main__":
    app.run()

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allwed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def uploads_file():
    if request.method == "POST":
        if "file" not in request.files:
            flash("File is not found.")
            return redirect(request.url)
        file = request.files["file"]
        input_name = request.form["text"]

        if file.filename == "":
            flash("File is not found.")
            return redirect(request.url)
        if file and allwed_file(file.filename):
            filename = secure_filename(file.filename)
            filename = os.path.splitext(os.path.basename(filename))[0] + ".txt"
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

            FILE_PATH = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            output = subprocess.check_output(
                ["sed", "1,2d", FILE_PATH], stderr=subprocess.PIPE
            )
            output = output.decode("utf8")

            with open(FILE_PATH, mode="w") as f:
                f.write(output)

            print("input_data: " + filename)
            print("input_name: " + input_name)

            neuro(filename, input_name)

            return redirect(url_for("uploaded_file", filename=filename))
    return render_template("index.html")


@app.route("/out/<filename>")
def uploaded_file(filename):
    # filename = "test.docx"
    return send_from_directory(
        UPLOAD_FOLDER,
        os.path.splitext(os.path.basename(filename))[0] + ".docx",
        as_attachment=True,
    )

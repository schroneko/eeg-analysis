import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from docx import Document

import app


class DependencyCompatibilityTests(unittest.TestCase):
    def test_upload_form(self):
        response = app.app.test_client().get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"multipart/form-data", response.data)

    def test_eeg_upload_produces_downloadable_word_report(self):
        repository = Path(app.__file__).parent
        with tempfile.TemporaryDirectory(dir=repository) as directory:
            root = Path(directory)
            output = root / "out"
            output.mkdir()
            shutil.copytree(repository / "npy", root / "npy")
            electrodes = [
                "Fp1-A1", "Fp2-A2", "F7-A1", "F8-A2", "C3-A1", "C4-A2",
                "T3-A1", "T4-A2", "T5-A1", "T6-A2", "O1-A1", "O2-A2",
            ]
            samples = np.random.default_rng(42).normal(size=(2000, len(electrodes)))
            fixture = io.StringIO()
            np.savetxt(
                fixture,
                np.column_stack([np.zeros(2000), np.zeros(2000), samples, np.zeros(2000), np.zeros(2000), np.zeros(2000)]),
                header="Synthetic EEG\n" + " ".join(electrodes),
                comments="",
            )
            with patch("os.getcwd", return_value=str(root)), patch.object(
                app, "UPLOAD_FOLDER", str(output)
            ), patch.dict(app.app.config, {"UPLOAD_FOLDER": str(output)}):
                client = app.app.test_client()
                response = client.post(
                    "/",
                    data={
                        "text": "Synthetic compatibility fixture",
                        "file": (io.BytesIO(fixture.getvalue().encode()), "fixture.m00"),
                    },
                    content_type="multipart/form-data",
                )
                self.assertEqual(response.status_code, 302)
                report_response = client.get(response.headers["Location"])
                self.assertEqual(report_response.status_code, 200)
                self.assertIn("attachment", report_response.headers["Content-Disposition"])
                document = Document(io.BytesIO(report_response.data))
                self.assertEqual(len(document.inline_shapes), 3)
                self.assertIn(
                    "Synthetic compatibility fixture",
                    "\n".join(paragraph.text for paragraph in document.paragraphs),
                )
                report_response.close()
            plt.close("all")


if __name__ == "__main__":
    unittest.main()

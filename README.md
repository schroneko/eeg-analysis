# EEG analysis

Use Python 3.11.15, as declared in `.python-version` and `runtime.txt`. The patched Flask and scientific dependencies require a newer runtime than the former Python 3.7 declaration.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m flask --app app run
```

Run from the repository directory so the existing `npy/`, `out/` and template paths resolve. Keep generated reports in a writable `out/` directory. The existing `Procfile` uses Gunicorn, now included in `requirements.txt`; `.venv/bin/gunicorn app:app` serves the same Flask application.

Compatibility was checked on Python 3.11.15 with the Flask upload page and a synthetic EEG calculation through DOCX generation with embedded graphs. This checks software compatibility, not clinical validity. Runtime declarations do not provision or update any hosted service.

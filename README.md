# Red Wine Quality — FastAPI stage in VS Code

This project loads your saved Colab pipeline and provides HTTP prediction endpoints. Training stays in Colab/Jupyter.

No trained model is included. The tests require your own Colab artifacts.

## 1. Save the completed Colab work

Run notebook sections 9–11 if you have not already:
- Save model.joblib, parameters, metadata, metrics, and the sample input/output.
- Download wine_quality_artifacts.zip.
- Use File → Download → Download .ipynb to keep your executed notebook with outputs.
- Keep screenshots of model comparison, test results, and the reload check.

If you change training/model settings, rerun all later notebook cells before downloading a new artifact bundle.

## 2. Open this starter in VS Code

Extract wine-quality-api-starter.zip. In VS Code choose File → Open Folder and select the extracted wine-quality-api-starter folder (the one containing main.py).

Extract your Colab ZIP directly into this project's artifacts folder. The paths must be artifacts/model.joblib and artifacts/metadata.json, not artifacts/artifacts/model.joblib.

Your project should contain:

    wine-quality-api-starter/
      main.py
      requirements.txt
      requirements-api.txt
      requirements-dev.txt
      verify_prediction.py
      tests/
        test_api.py
      artifacts/
        model.joblib
        metadata.json
        parameters.json
        metrics.json
        requirements-model.txt
        sample_request.json
        expected_prediction.json
        cv_results.csv
        test_predictions.csv

## 3. Match the Python version to Colab

Open Terminal → New Terminal in VS Code. These commands use Windows PowerShell and should run from the project folder.

Read the Python minor version recorded during training:

    $colabPython = (Get-Content .\artifacts\metadata.json -Raw | ConvertFrom-Json).python_major_minor
    $colabPython

List installed Python versions:

    py --list

If the matching version is missing, install it from https://www.python.org/downloads/ and reopen the terminal. Use a standard 64-bit Python installation.

Create a virtual environment with the matching version:

    py "-$colabPython" -m venv .venv

Then in VS Code use Ctrl+Shift+P → Python: Select Interpreter → choose the interpreter inside .venv.
Install the Microsoft Python extension if the interpreter command is unavailable.

## 4. Install the dependencies

These commands call the virtual environment directly:

    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
    .\.venv\Scripts\python.exe -m pip check

requirements.txt includes the exact model dependencies exported by Colab. The API and test libraries are pinned separately. Keep these model pins when moving to Docker.

## 5. Run the automated checks

    .\.venv\Scripts\python.exe -m pytest -q

These checks cover model loading, prediction agreement with Colab, JSON field order, and invalid requests.

A passing run is required before proceeding. Do not replace your saved model with another model to make a test pass.

If startup reports a version mismatch, recreate the virtual environment with the correct Python and install the exported requirements. If a file is missing, check the artifacts folder layout.

## 6. Start the API

    .\.venv\Scripts\python.exe -m uvicorn main:app --reload

Keep this terminal running. Wait for Application startup complete.

Open http://127.0.0.1:8000/docs in your browser. FastAPI supplies this interactive documentation.

Available endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | /health | Confirm the application started and the model loaded |
| GET | /model-info | Inspect the selected model and input feature names |
| POST | /predict | Submit 11 measurements and return predicted_quality |

The prediction request uses lowercase ph. All 11 measurements are required. Values must be JSON numbers; missing keys, nulls, unknown keys, and invalid physical values are rejected with status 422. The training pipeline can impute during training; this API deliberately asks callers for complete measurements.

## 7. Make a prediction in the browser

1. In /docs, expand GET /health, select Try it out, then Execute. Expect status 200 and model_loaded: true.
2. Open artifacts/sample_request.json in VS Code and copy the complete JSON object.
3. In /docs, expand POST /predict and select Try it out.
4. Replace the example request body with your saved JSON.
5. Select Execute. Expect status 200 and a numeric predicted_quality.
6. Compare it with artifacts/expected_prediction.json. The prediction should match apart from insignificant floating-point differences.

The app returns the original floating-point score; it does not round the score or retrain the model.

## 8. Verify the running HTTP endpoint

Open a second VS Code terminal while the server remains running:

    .\.venv\Scripts\python.exe .\verify_prediction.py

Expect a PASS message comparing your API prediction to Colab.

You can also send the saved request from PowerShell:

    $wineBody = Get-Content .\artifacts\sample_request.json -Raw
    Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body $wineBody

To demonstrate validation, remove the alcohol key from the JSON in /docs and execute again. Expect 422. Restore the complete saved request afterward.

## Evidence to keep

Save screenshots of:
- The passing pytest run.
- GET /health returning 200.
- POST /predict returning 200.
- The running-endpoint prediction comparison reporting PASS.
- An invalid request returning 422.

This completes the local FastAPI prediction stage. Next, add a Dockerfile, run the same endpoint checks through Docker, then create the GitHub repository and CI pipeline.

## How the code works

The lifespan function loads the model once during startup and checks the exported Python/package versions. The request schema validates 11 measurements. The prediction route constructs a pandas DataFrame in the feature order stored by Colab and calls the already-fitted pipeline. The model and its learned preprocessing are reused together.

References: https://fastapi.tiangolo.com/tutorial/first-steps/ , https://fastapi.tiangolo.com/advanced/events/ , https://fastapi.tiangolo.com/tutorial/testing/ , https://scikit-learn.org/stable/model_persistence.html

## Assignment results

- Case study: red-wine quality regression using the Kaggle Wine Quality dataset.
- Training environment: Google Colab; the notebook source is in notebooks/. Keep your executed Colab notebook as assignment evidence.
- Selected model: tuned Random Forest, saved together with preprocessing in artifacts/model.joblib.
- Test MAE: 0.4709; RMSE: 0.6187; R2: 0.4595.
- Local FastAPI verification: 11 automated tests passed; the running endpoint reproduced the saved Colab prediction of 5.3.

### Docker commands

Run these from the project folder in your normal VS Code terminal with Docker Desktop running:

    docker build -t wine-quality-api .
    docker run --rm -d --name wine-quality-api -p 8001:8000 wine-quality-api
    .\.venv\Scripts\python.exe verify_prediction.py --url http://127.0.0.1:8001

The local FastAPI server uses port8000; the Docker example uses port8001.

### GitHub Actions

.github/workflows/ci.yml installs dependencies, runs API tests, builds the Docker image, starts the container, checks its health, and verifies a real prediction against the saved Colab value. It runs on pushes, pull requests, or manual workflow dispatch.

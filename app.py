import logging
import os
import tempfile
from datetime import datetime
from io import BytesIO
from typing import List, Tuple
from zipfile import ZipFile

import openpyxl
import pandas as pd
import requests
from flask import Flask, jsonify, make_response, request
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from werkzeug.exceptions import HTTPException

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("name_classifier.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Constants
MODEL_FILE = "name_classifier.joblib"
TRAINING_DATA_URLS = [
    "https://raw.githubusercontent.com/ErnestoAizenberg/kz-ru-name-detector/refs/heads/main/ru_names_1.xlsx",
    "https://raw.githubusercontent.com/ErnestoAizenberg/kz-ru-name-detector/refs/heads/main/kz_names_1.xlsx",
    "https://raw.githubusercontent.com/ErnestoAizenberg/kz-ru-name-detector/refs/heads/main/kz_names_2.xlsx",
]


class ModelTrainingError(Exception):
    """Custom exception for model training failures"""

    pass


class FileProcessingError(Exception):
    """Custom exception for file processing failures"""

    pass


def setup_model() -> Pipeline:
    """Initialize or load the classification model"""
    try:
        if not os.path.exists(MODEL_FILE):
            logger.info("Model not found. Training new model...")
            model = train_model()
            dump(model, MODEL_FILE)
            logger.info(f"Model successfully trained and saved to {MODEL_FILE}")
        else:
            logger.info(f"Loading model from {MODEL_FILE}")
            model = load(MODEL_FILE)
        return model
    except Exception as e:
        logger.error(f"Failed to setup model: {str(e)}")
        raise ModelTrainingError("Could not initialize the classification model")


def download_training_data() -> None:
    """Download training data files"""
    try:
        for url in TRAINING_DATA_URLS:
            filename = url.split("/")[-1]
            logger.info(f"Downloading training data: {filename}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            with open(filename, "wb") as f:
                f.write(response.content)
            logger.info(f"Successfully downloaded {filename}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download training data: {str(e)}")
        raise ModelTrainingError("Could not download training data")


def df_to_string_list(df: pd.DataFrame) -> List[str]:
    """Convert DataFrame rows to list of strings"""
    try:
        return [
            " ".join(str(cell).strip() if pd.notna(cell) else "" for cell in row)
            for row in df.values.tolist()
        ]
    except Exception as e:
        logger.error(f"DataFrame conversion failed: {str(e)}")
        raise ModelTrainingError("Could not process training data")


def train_model() -> Pipeline:
    """Train and return a new classification model"""
    try:
        download_training_data()

        # Load and process training data
        df_ru = pd.read_excel("ru_names_1.xlsx", usecols=[0, 1, 2], header=None)
        df_kz1 = pd.read_excel("kz_names_1.xlsx", usecols=[0, 1, 2], header=None)
        df_kz2 = pd.read_excel("kz_names_2.xlsx", usecols=[3], header=None)

        list_ru = df_to_string_list(df_ru)
        list_kz = df_to_string_list(df_kz1) + df_to_string_list(df_kz2)

        # Create training DataFrame
        df = pd.concat(
            [
                pd.DataFrame({"name": list_kz, "label": "kz"}),
                pd.DataFrame({"name": list_ru, "label": "ru"}),
            ]
        )

        # Train model
        model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 3))),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            df["name"], df["label"], test_size=0.2, random_state=42
        )

        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        logger.info(f"Model trained with accuracy: {accuracy:.2f}")

        return model
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise ModelTrainingError("Model training process failed")


def join_cols_to_full_name(row: tuple, usecols: List[int]) -> str:
    """Combine specified columns into a full name string"""
    try:
        return " ".join(
            str(row[col]).strip() for col in usecols if row[col] is not None
        ).strip()
    except (IndexError, TypeError) as e:
        logger.warning(f"Failed to join name parts: {str(e)}, row: {row}")
        return ""


def validate_name_columns(columns_str: str, max_columns: int = 10) -> List[int]:
    """Validate and parse name columns input"""
    try:
        columns = [int(col.strip()) for col in columns_str.split(",")]
        if not columns:
            raise ValueError("No columns specified")
        if any(col < 0 for col in columns):
            raise ValueError("Column numbers cannot be negative")
        if len(columns) > max_columns:
            raise ValueError(f"Maximum {max_columns} columns allowed")
        return columns
    except ValueError as e:
        logger.error(f"Invalid name columns input: {columns_str} - {str(e)}")
        raise


def has_kazakh_letters(name):
    kazakh_letters = {
        "Ә",
        "ә",
        "Ғ",
        "ғ",
        "Қ",
        "қ",
        "Ң",
        "ң",
        "Ө",
        "ө",
        "Ұ",
        "ұ",
        "Ү",
        "ү",
        "Һ",
        "һ",
        "І",
        "і",
    }
    return any(char in kazakh_letters for char in name)


def process_uploaded_file(
    file_stream, name_columns: List[int]
) -> Tuple[BytesIO, int, int]:
    """Process the uploaded Excel file and return results"""
    try:
        with (
            tempfile.NamedTemporaryFile(suffix=".xlsx") as tmp_source,
            tempfile.NamedTemporaryFile(suffix=".xlsx") as tmp_kz,
            tempfile.NamedTemporaryFile(suffix=".xlsx") as tmp_ru,
        ):
            # Save uploaded file
            file_stream.save(tmp_source.name)

            # Process the file
            try:
                wb_source = openpyxl.load_workbook(tmp_source.name)
                ws_source = wb_source.active
            except Exception as e:
                logger.error(f"Invalid Excel file: {str(e)}")
                raise FileProcessingError("The uploaded file is not a valid Excel file")

            # Prepare output files
            wb_kz = openpyxl.Workbook()
            ws_kz = wb_kz.active
            wb_ru = openpyxl.Workbook()
            ws_ru = wb_ru.active

            # Copy headers if they exist
            if ws_source.max_row > 0:
                headers = [cell.value for cell in ws_source[1]]
                ws_kz.append(headers)
                ws_ru.append(headers)

            # Process rows
            kz_count = 0
            ru_count = 0
            processed_rows = 0

            for row in ws_source.iter_rows(min_row=2, values_only=True):
                processed_rows += 1
                try:
                    full_name = join_cols_to_full_name(row, name_columns)
                    if not full_name:
                        continue

                    if has_kazakh_letters(full_name):
                        ws_kz.append(row)
                        kz_count += 1
                        continue

                    prediction = model.predict([full_name])[0]
                    if prediction == "kz":
                        ws_kz.append(row)
                        kz_count += 1
                    elif prediction == "ru":
                        ws_ru.append(row)
                        ru_count += 1
                    else:
                        logger.warning(
                            f"Unexpected prediction for '{full_name}': {prediction}"
                        )
                except Exception as e:
                    logger.warning(f"Error processing row {processed_rows}: {str(e)}")
                    continue

            logger.info(
                f"Processed {processed_rows} rows. Results: KZ={kz_count}, RU={ru_count}"
            )

            # Save results
            wb_kz.save(tmp_kz.name)
            wb_ru.save(tmp_ru.name)

            # Create zip archive
            zip_buffer = BytesIO()
            with ZipFile(zip_buffer, "w") as zip_file:
                zip_file.write(tmp_kz.name, "kz_names.xlsx")
                zip_file.write(tmp_ru.name, "ru_names.xlsx")

            zip_buffer.seek(0)
            return zip_buffer, kz_count, ru_count

    except Exception as e:
        logger.error(f"File processing failed: {str(e)}")
        raise FileProcessingError("Could not process the uploaded file")


# Error handlers
@app.errorhandler(HTTPException)
def handle_http_error(e):
    logger.warning(f"HTTP error {e.code}: {e.description}")
    return jsonify({"error": e.name, "message": e.description}), e.code


@app.errorhandler(ModelTrainingError)
def handle_model_error(e):
    logger.error(f"Model error: {str(e)}")
    return jsonify({"error": "Model Error", "message": str(e)}), 500


@app.errorhandler(FileProcessingError)
def handle_file_error(e):
    logger.error(f"File processing error: {str(e)}")
    return jsonify({"error": "File Processing Error", "message": str(e)}), 400


@app.errorhandler(Exception)
def handle_unexpected_error(e):
    logger.critical(f"Unexpected error: {str(e)}", exc_info=True)
    return (
        jsonify(
            {
                "error": "Internal Server Error",
                "message": "An unexpected error occurred",
            }
        ),
        500,
    )


# Initialize model
try:
    model = setup_model()
except ModelTrainingError:
    logger.critical("Application failed to start due to model initialization error")
    raise


@app.route("/process_names", methods=["POST"])
def process_names():
    """API endpoint for processing name files"""
    try:
        logger.info("Name processing request received")

        # Validate request
        if "file" not in request.files:
            logger.warning("No file uploaded")
            raise FileProcessingError("No file uploaded")

        file = request.files["file"]
        if file.filename == "":
            logger.warning("Empty filename")
            raise FileProcessingError("No file selected")

        if not file.filename.lower().endswith((".xlsx", ".xls")):
            logger.warning(f"Invalid file type: {file.filename}")
            raise FileProcessingError("Only Excel files (.xlsx, .xls) are supported")

        # Get name columns parameter
        name_columns_str = request.form.get("name_columns", "0,1,2")
        try:
            name_columns = validate_name_columns(name_columns_str)
        except ValueError as e:
            raise FileProcessingError(str(e))

        # Process file
        try:
            zip_buffer, kz_count, ru_count = process_uploaded_file(file, name_columns)
        except Exception as e:
            logger.error(f"File processing failed: {str(e)}")
            raise FileProcessingError("Could not process the file")

        # Prepare response
        response = make_response(zip_buffer.getvalue())
        response.headers.set("Content-Type", "application/zip")
        response.headers.set(
            "Content-Disposition",
            "attachment",
            filename=f"classified_names_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        )

        logger.info("Successfully processed file")
        return response

    except FileProcessingError:
        raise  # Will be handled by the error handler
    except Exception as e:
        logger.error(f"Unexpected processing error: {str(e)}")
        raise FileProcessingError("An unexpected error occurred during processing")


@app.route("/classify_name", methods=["POST"])
def classify_name():
    """Endpoint for classifying a single name"""
    try:
        data = request.get_json()
        name = data.get("name", "").strip()

        if not name:
            return jsonify({"error": "Name is required"}), 400

        # First check for Kazakh letters
        if has_kazakh_letters(name):
            return jsonify(
                {
                    "name": name,
                    "classification": "kz",
                    "method": "kazakh_letters_check",
                    "probability": 1.0,
                }
            )

        # Use model for prediction
        prediction = model.predict([name])[0]
        proba = model.predict_proba([name])[0]
        probability = proba[0] if prediction == "kz" else proba[1]

        return jsonify(
            {
                "name": name,
                "classification": prediction,
                "method": "model_prediction",
                "probability": float(probability),
            }
        )

    except Exception as e:
        logger.error(f"Error classifying name: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/")
def index():
    """Main page with upload form"""
    logger.info("Serving index page")
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Name Classifier (KZ/RU)</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                color: #333;
            }
            h1 {
                color: #2c3e50;
                text-align: center;
            }
            .form-container {
                background: #f9f9f9;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .form-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            input[type="file"] {
                width: 100%;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                background: white;
            }
            input[type="text"] {
                width: 100%;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            input[type="submit"] {
                background: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                display: block;
                margin: 20px auto 0;
            }
            input[type="submit"]:hover {
                background: #2980b9;
            }
            .submit-btn {
                background: #2ecc71;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                cursor: pointer;
            }
            .submit-btn:hover {
                background: #27ae60;
            }
            .result-kz {
                background-color: #e8f8f5;
                border: 1px solid #2ecc71;
            }
            .result-ru {
                background-color: #fef9e7;
                border: 1px solid #f39c12;
            }
            .probability {
                margin-top: 10px;
                height: 20px;
                background: #ecf0f1;
                border-radius: 3px;
                overflow: hidden;
            }
            .probability-bar {
                height: 100%;
                background: #3498db;
                width: 0%;
                transition: width 0.5s;
            }
            small {
                color: #7f8c8d;
                font-size: 0.9em;
            }
            .file-info {
                margin-top: 5px;
                font-size: 0.9em;
                color: #7f8c8d;
            }
            @media (max-width: 600px) {
                body {
                    padding: 10px;
                }
                .form-container {
                    padding: 15px;
                }
            }
        </style>
    </head>
    <body>
        <h1>Name Classifier (KZ/RU)</h1>
        <div class="form-container">
            <form action="/process_names" method="post" enctype="multipart/form-data" id="uploadForm">
                <div class="form-group">
                    <label for="file">Upload Excel file with names:</label>
                    <input type="file" id="file" name="file" accept=".xlsx,.xls" required>
                    <div class="file-info">Accepted formats: .xlsx, .xls</div>
                </div>

                <div class="form-group">
                    <label for="name_columns">Columns containing name parts (0-based, comma-separated):</label>
                    <input type="text" id="name_columns" name="name_columns" value="0,1,2" required>
                    <small>Example: For first name in column 0 and last name in column 1, use "0,1"</small>
                </div>

                <input type="submit" value="Process" id="submitBtn">
            </form>
        </div>

        <div class="form-container" style="margin-top: 30px;">
            <h3 style="text-align: center; margin-bottom: 15px;">Test Single Name</h3>
            <form id="singleNameForm" onsubmit="checkSingleName(event)">
                <div class="form-group">
                    <label for="test_name">Enter a name to classify:</label>
                    <input type="text" id="test_name" name="test_name" required>
                </div>
                <input type="submit" value="Check" class="submit-btn">
            </form>
            <div id="result" style="margin-top: 20px; padding: 15px; border-radius: 5px; display: none;"></div>
        </div>
        <script>
            // Basic form validation
            document.getElementById('uploadForm').addEventListener('submit', function(e) {
                const fileInput = document.getElementById('file');
                const columnsInput = document.getElementById('name_columns');
                const submitBtn = document.getElementById('submitBtn');

                // Validate file extension
                const fileName = fileInput.value.toLowerCase();
                if (!fileName.endsWith('.xlsx') && !fileName.endsWith('.xls')) {
                    alert('Please upload an Excel file (.xlsx or .xls)');
                    e.preventDefault();
                    return;
                }

                // Validate columns format
                if (!/^\d+(,\d+)*$/.test(columnsInput.value)) {
                    alert('Please enter valid column numbers (e.g. "0,1,2")');
                    e.preventDefault();
                    return;
                }

                // Disable button to prevent double submission
                submitBtn.disabled = true;
                submitBtn.value = 'Processing...';
            });
            // Handle single name check
            function checkSingleName(event) {
                event.preventDefault();
                const name = document.getElementById('test_name').value.trim();
                if (!name) return;

                fetch('/classify_name', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({name: name})
                })
                .then(response => response.json())
                .then(data => {
                    const resultDiv = document.getElementById('result');
                    resultDiv.style.display = 'block';

                    if (data.classification === 'kz') {
                        resultDiv.className = 'result result-kz';
                        resultDiv.innerHTML = `
                            <h4>Result for: ${data.name}</h4>
                            <p><strong>Classification:</strong> Kazakh (KZ)</p>
                            ${data.method === 'kazakh_letters_check' ?
                              '<p>Detected Kazakh letters</p>' :
                              `<p>Model prediction (${(data.probability * 100).toFixed(1)}% confidence)</p>`
                            }
                        `;
                    } else {
                        resultDiv.className = 'result result-ru';
                        resultDiv.innerHTML = `
                            <h4>Result for: ${data.name}</h4>
                            <p><strong>Classification:</strong> Russian (RU)</p>
                            <p>Model prediction (${(data.probability * 100).toFixed(1)}% confidence)</p>
                            <div class="probability">
                                <div class="probability-bar" style="width: ${data.probability * 100}%"></div>
                            </div>
                        `;
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('An error occurred while processing the name');
                });
            }
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000)
    except Exception as e:
        logger.critical(f"Application failed to start: {str(e)}")
        raise

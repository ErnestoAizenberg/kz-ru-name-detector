import logging
import os
import unittest
from io import BytesIO
from unittest.mock import patch

import openpyxl
from app import app, process_uploaded_file, setup_model

# Configure detailed test logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    filename="test_name_classifier.log",
    filemode='w'  # Overwrite log file for each test run
)
logger = logging.getLogger(__name__)


class TestNameClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize the test environment"""
        logger.debug("Initializing test environment")
        cls.app = app.test_client()
        cls.app.testing = True

        # Create a sample test file
        test_file = "kz_names_1.xlsx"
        cls.create_test_file(test_file)
        logger.debug(f"Created test file: {test_file} with sample data")
        logger.info("Test setup completed successfully")

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests"""
        test_file = "kz_names_1.xlsx"
        if os.path.exists(test_file):
            os.remove(test_file)
            logger.debug(f"Removed test file: {test_file}")
        logger.info("Test cleanup completed")

    @staticmethod
    def create_test_file(filename):
        """Create a sample Excel file for testing"""
        logger.debug(f"Creating test file {filename} with sample data")
        wb = openpyxl.Workbook()
        ws = wb.active
        headers = ["First Name", "Last Name", "Middle Name", "Address", "Phone"]
        ws.append(headers)
        logger.debug(f"Added headers: {headers}")

        test_data = [
            ["Айдар", "Мухамеджанов", "Сергеевич", "Алматы, ул. Абая 1", "7771234567"],
            ["Иван", "Петров", "Иванович", "Москва, ул. Ленина 5", "4951234567"],
            ["Алия", "Қарақыз", "", "Астана, пр. Республики 10", "7172123456"],
            ["Сергей", "Смирнов", "Петрович", "Санкт-Петербург, Невский пр. 20", "8123456789"],
        ]

        for row in test_data:
            ws.append(row)
            logger.debug(f"Added test row: {row}")

        wb.save(filename)
        logger.debug(f"Test file saved: {filename}")

    def test_01_setup_model(self):
        """Test model setup"""
        logger.debug("Starting model setup test")
        model = setup_model()
        self.assertIsNotNone(model, "Model should be initialized")
        logger.debug("Model initialized successfully")
        logger.info("Model setup test passed")

    def test_02_process_file_no_filters(self):
        """Test file processing with no filters"""
        logger.debug("Starting file processing test with no filters")
        with open("kz_names_1.xlsx", "rb") as f:
            buffer, count = process_uploaded_file(
                f,
                name_columns=[0, 1, 2],
                country_filter=None,
                address_columns=None,
                search_value=None,
            )

        logger.debug(f"Processed {count} records with no filters")
        self.assertEqual(count, 4, "Should process all 4 records")
        logger.info("No filters test passed - all records processed")

    def test_03_process_file_kz_filter(self):
        """Test file processing with KZ filter"""
        logger.debug("Starting file processing test with KZ filter")
        with open("kz_names_1.xlsx", "rb") as f:
            buffer, count = process_uploaded_file(
                f,
                name_columns=[0, 1, 2],
                country_filter="kz",
                address_columns=None,
                search_value=None,
            )

        logger.debug(f"Found {count} records matching KZ filter")
        self.assertGreaterEqual(count, 2, "Should find at least 2 KZ names")
        logger.info(f"KZ filter test passed - found {count} records")

    def test_04_process_file_ru_filter(self):
        """Test file processing with RU filter"""
        logger.debug("Starting file processing test with RU filter")
        with open("kz_names_1.xlsx", "rb") as f:
            buffer, count = process_uploaded_file(
                f,
                name_columns=[0, 1, 2],
                country_filter="ru",
                address_columns=None,
                search_value=None,
            )

        logger.debug(f"Found {count} records matching RU filter")
        self.assertGreaterEqual(count, 2, "Should find at least 2 RU names")
        logger.info(f"RU filter test passed - found {count} records")

    def test_05_process_file_address_search(self):
        """Test file processing with address search"""
        search_term = "Алматы"
        logger.debug(f"Starting address search test for: '{search_term}'")
        with open("kz_names_1.xlsx", "rb") as f:
            buffer, count = process_uploaded_file(
                f,
                name_columns=[0, 1, 2],
                country_filter=None,
                address_columns=[3],
                search_value=search_term,
            )

        logger.debug(f"Found {count} records matching address search: '{search_term}'")
        self.assertEqual(count, 1, f"Should find 1 record with '{search_term}' in address")
        logger.info("Address search test passed - found matching record")

    def test_06_process_file_combined_filters(self):
        """Test file processing with combined filters"""
        country_filter = "kz"
        search_term = "Астана"
        logger.debug(f"Starting combined filter test (country: {country_filter}, search: '{search_term}')")
        with open("kz_names_1.xlsx", "rb") as f:
            buffer, count = process_uploaded_file(
                f,
                name_columns=[0, 1, 2],
                country_filter=country_filter,
                address_columns=[3],
                search_value=search_term,
            )

        logger.debug(f"Found {count} records matching combined filters")
        self.assertEqual(count, 1, f"Should find 1 {country_filter.upper()} record with '{search_term}' in address")
        logger.info("Combined filters test passed - found exact match")

    def test_07_api_endpoint(self):
        """Test the API endpoint with file upload"""
        logger.debug("Starting API endpoint test")
        with open("kz_names_1.xlsx", "rb") as f:
            data = {
                "name_columns": "0,1,2",
                "country_filter": "kz",
                "address_columns": "3",
                "search_value": "Астана",
            }
            logger.debug(f"Sending request to API with data: {data}")
            response = self.app.post(
                "/process_names",
                data=data,
                content_type="multipart/form-data",
                buffered=True,
                headers={"Content-Length": str(os.path.getsize("kz_names_1.xlsx"))},
                input_stream=lambda: f,
            )

        logger.debug(f"API response status code: {response.status_code}")
        self.assertEqual(response.status_code, 200, "API should return 200 OK")
        logger.info("API endpoint test passed")

    @patch("name_classifier.request")
    def test_08_classify_name_endpoint(self, mock_request):
        """Test the name classification endpoint"""
        test_name = "Айдар"
        logger.debug(f"Starting name classification test for name: '{test_name}'")
        mock_request.get_json.return_value = {"name": test_name}

        with app.test_request_context():
            response = app.classify_name()

        logger.debug(f"Classification response status code: {response.status_code}")
        self.assertEqual(response.status_code, 200, "Should return 200 OK")
        logger.info("Name classification endpoint test passed")

    def test_09_invalid_file_handling(self):
        """Test handling of invalid files"""
        logger.debug("Starting invalid file handling test")
        with self.assertRaises(Exception) as context:
            process_uploaded_file(
                BytesIO(b"invalid file content"),
                name_columns=[0, 1, 2],
                country_filter=None,
                address_columns=None,
                search_value=None,
            )

        logger.debug(f"Correctly raised exception for invalid file: {str(context.exception)}")
        logger.info("Invalid file handling test passed")


if __name__ == "__main__":
    logger.info("Starting test suite execution")
    unittest.main()

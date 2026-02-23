import sys
from pathlib import Path
import logging

def get_valid_path(path_str, default="./data/"):
    if not path_str:
        logging.warning("Source path directory is mandatory. Setting it to default.")
        path_str = default

    path = Path(path_str)
    if not path.is_dir():
        logging.error(f"{path} isn't a directory")
        sys.exit(1)
    return path

def validate_file_exists(file_path):
    if not file_path.is_file():
        logging.error(f"{file_path} doesn't exist in {file_path.parent}")
        sys.exit(1)

def run_importer(importer_factory, base_cls, backend: str, file_name: str, base_path: str = "./data/"):
    # Create the importer class dynamically using the backend and base class
    ImporterClass = importer_factory(base_cls, backend)

    # Validate and construct the full file path
    data_dir = get_valid_path(base_path)
    data_file = data_dir / file_name
    validate_file_exists(data_file)

    # Instantiate and run
    importer = ImporterClass()
    logging.info(f"Importing {file_name} records using backend '{backend}'...")
    importer.import_data(str(data_file))
    importer.close()

def run_updater(importer_factory, base_cls, backend: str):
    # Create the importer class dynamically using the backend and base class
    ImporterClass = importer_factory(base_cls, backend)

    # Instantiate and run update logic
    updater = ImporterClass()
    logging.info(f"Applying updates to the graph using backend '{backend}'...")
    updater.apply_updates()
    updater.close()
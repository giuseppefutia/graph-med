import argparse
from util.logger import setup_logging
from util.cli_utils import run_importer, run_updater
from importer.neo4j_importer import Neo4jBaseImporter

def run_backend_importer(
    importer_factory_func,
    description,
    file_help,
    default_base_path="./data/",
    require_file=True,
):
    setup_logging()

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--backend", choices=["neo4j"], required=True,
                        help="Which importer backend to use")
    
    # Only require --file if the importing flow needs it
    if require_file:
        parser.add_argument("--file", required=True, help=file_help)
    else:
        parser.add_argument("--file", help=file_help)

    parser.add_argument("--base_path", default=default_base_path,
                        help="Base directory where the file is located")

    args = parser.parse_args()

    backend_map = {
        "neo4j": Neo4jBaseImporter,
    }

    base_cls = backend_map.get(args.backend)
    if base_cls is None:
        raise ValueError(f"Unsupported backend: {args.backend}")

    if require_file:
        run_importer(importer_factory_func, base_cls, args.backend, args.file, base_path=args.base_path)
    else:
        run_updater(importer_factory_func, base_cls, args.backend)

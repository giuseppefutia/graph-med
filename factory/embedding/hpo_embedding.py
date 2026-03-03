from factory.embedding.base_embedding import embedding_factory

_HPO_NODE_SPECS = [
    {
        "label": "HpoPhenotype",
        "id_prop": "id",
        "text_prop": "label",
        "embed_prop": "embedding_label",
        "index_name": "hpo_phenotype_embedding",
        "dim": 3584,
        "similarity": "cosine",
        "log_tag": "HPO-Phen",
    },
    {
        "label": "HpoDisease",
        "id_prop": "id",
        "text_prop": "label",
        "embed_prop": "embedding_label",
        "index_name": "hpo_disease_embedding",
        "dim": 3584,
        "similarity": "cosine",
        "log_tag": "HPO-Disease",
    },
]


def hpo_embedding_importer_factory(
    base_importer_cls, backend: str, config_path: str = "config.ini"
):
    return embedding_factory(base_importer_cls, backend, _HPO_NODE_SPECS, config_path)


if __name__ == "__main__":
    from util.cli_entry import run_backend_importer

    run_backend_importer(
        hpo_embedding_importer_factory,
        description="Run HPO Embedding.",
        file_help="No file needed for HPO embedding.",
        default_base_path="./data/",
        require_file=False,
    )

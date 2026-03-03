from factory.embedding.base_embedding import embedding_factory

_ICD_NODE_SPECS = [
    {
        "label": "IcdDisease",
        "id_prop": "id",
        "text_prop": "label",
        "embed_prop": "embedding_label",
        "index_name": "icd_disease_embedding",
        "dim": 3584,
        "similarity": "cosine",
        "log_tag": "ICD-Disease",
    },
    {
        "label": "IcdChapter",
        "id_prop": "id",
        "text_prop": "chapterName",
        "embed_prop": "embedding_label",
        "index_name": "icd_chapter_embedding",
        "dim": 3584,
        "similarity": "cosine",
        "log_tag": "ICD-Chapter",
    },
    {
        "label": "IcdGroup",
        "id_prop": "id",
        "text_prop": "groupName",
        "embed_prop": "embedding_label",
        "index_name": "icd_group_embedding",
        "dim": 3584,
        "similarity": "cosine",
        "log_tag": "ICD-Group",
    },
]


def icd_embedding_importer_factory(
    base_importer_cls, backend: str, config_path: str = "config.ini"
):
    return embedding_factory(base_importer_cls, backend, _ICD_NODE_SPECS, config_path)


if __name__ == "__main__":
    from util.cli_entry import run_backend_importer

    run_backend_importer(
        icd_embedding_importer_factory,
        description="Run ICD Embedding.",
        file_help="No file needed for ICD embedding.",
        default_base_path="./data/",
        require_file=False,
    )

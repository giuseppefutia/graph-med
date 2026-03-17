import os
import configparser
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_config(path: str) -> str:
    """Return *path* if it exists, otherwise look in the project root."""
    if os.path.isfile(path):
        return path
    fallback = str(_PROJECT_ROOT / os.path.basename(path))
    if os.path.isfile(fallback):
        return fallback
    return path  # let the caller raise FileNotFoundError


def load_config(env_section: Optional[str] = None, path: str = "config.ini") -> str:
    """
    Return the `uri` from a section in config.ini.

    Falls back to API_ENV="api" when env_section isn't provided.
    Raises:
      - FileNotFoundError if the file can't be read
      - KeyError if the section is missing
      - ValueError if `uri` is missing/empty
    """
    cfg = configparser.ConfigParser()
    path = _resolve_config(path)
    if not cfg.read(path):
        raise FileNotFoundError(f"Couldn't find {path} in the current directory.")

    section = env_section or os.getenv("API_ENV", "api")
    if section not in cfg:
        raise KeyError(f"Section [{section}] not found in {path}.")

    uri = cfg.get(section, "uri", fallback="").strip().rstrip("/")
    if not uri:
        raise ValueError(f"[{section}] uri is empty in {path}.")

    return uri


def load_config_api(service: Optional[str] = None, path: str = "config.ini") -> str:
    """
    Convenience wrapper to fetch URIs by logical service name.

    Services:
      - "chat"      -> [chat-api]
      - "llm"       -> [open-api]
      - "embedding" -> [embedding-api]
      - "neo4j"     -> [neo4j]

    Default service comes from API_SERVICE="chat".
    """
    svc = (service or os.getenv("API_SERVICE", "chat")).lower()
    section_map = {
        "chat": "chat-api",
        "llm": "chat-api",
        "embedding": "embedding-api",
        "neo4j": "neo4j",
        "gnn": "gnn-api",
        "gretriever": "gretriever-api",
    }
    try:
        section = section_map[svc]
    except KeyError:
        raise ValueError(f'Unknown service "{svc}". Use one of: {", ".join(section_map)}.')
    return load_config(section, path=path)


def load_neo4j_config(path: str = "config.ini") -> dict:
    """Return Neo4j connection params from the [neo4j] section of config.ini."""
    cfg = configparser.ConfigParser()
    path = _resolve_config(path)
    if not cfg.read(path):
        raise FileNotFoundError(f"Couldn't find {path} in the current directory.")
    return {
        "url":      cfg.get("neo4j", "uri",      fallback="bolt://localhost:7687"),
        "username": cfg.get("neo4j", "user",     fallback="neo4j"),
        "password": cfg.get("neo4j", "password", fallback="password"),
        "database": cfg.get("neo4j", "database", fallback="neo4j"),
    }


if __name__ == "__main__":
    # Example usage
    for s in ("chat", "llm", "embedding", "neo4j"):
        try:
            print(f"{s}:", load_config_api(s))
        except Exception as e:
            print(f"{s} error:", e)

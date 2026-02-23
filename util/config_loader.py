import os
import configparser
from typing import Optional


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
        "llm": "open-api",
        "embedding": "embedding-api",
        "neo4j": "neo4j",
    }
    try:
        section = section_map[svc]
    except KeyError:
        raise ValueError(f'Unknown service "{svc}". Use one of: {", ".join(section_map)}.')
    return load_config(section, path=path)


if __name__ == "__main__":
    # Example usage
    for s in ("chat", "llm", "embedding", "neo4j"):
        try:
            print(f"{s}:", load_config_api(s))
        except Exception as e:
            print(f"{s} error:", e)

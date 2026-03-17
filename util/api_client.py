from __future__ import annotations
import json, urllib.request, urllib.parse
from typing import Any, Dict

def build_url(base_url: str, path: str, query: dict | None = None) -> str:
    """
    Join base_url with path and optional query dict.
    """
    path = "/" + path.lstrip("/")
    url = base_url.rstrip("/") + path
    if query:
        url += "?" + urllib.parse.urlencode(query, doseq=True)
    return url

def request(
    method: str,
    url: str,
    json_body: dict | None = None,
    headers: dict | None = None,
    timeout: int = 10,
) -> tuple[Any, int, Dict[str, str]]:
    """
    Minimal HTTP client using urllib.

    Returns: (body, status_code, headers)
      - body is JSON-decoded (dict/list) if Content-Type is application/json,
        otherwise a str.
    Raises:
      - RuntimeError on HTTP/connection errors with readable details.
    """
    method = method.upper()
    hdrs = {"Accept": "application/json"}
    data = None

    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        hdrs["Content-Type"] = "application/json"

    if headers:
        hdrs.update(headers)

    req = urllib.request.Request(url=url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body_bytes = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            if "application/json" in ctype:
                body = json.loads(body_bytes.decode("utf-8"))
            else:
                body = body_bytes.decode("utf-8")
            return body, resp.status, dict(resp.headers)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code} {e.reason}: {detail}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"Connection error: {e.reason}") from None

class ApiClient:
    """
    Tiny convenience wrapper around request() with a base URL

    Example:
        api = ApiClient("http://127.0.0.1:8000")
        body, status, _ = api.get("/health")
    """
    def __init__(self, base_url: str, auth_token: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.auth_headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}

    def _url(self, path: str, query: dict | None = None) -> str:
        return build_url(self.base_url, path, query)

    def get(
        self,
        path: str,
        query: dict | None = None,
        headers: dict | None = None,
        timeout: int = 10,
    ) -> tuple[Any, int, Dict[str, str]]:
        hdrs = dict(self.auth_headers)
        if headers:
            hdrs.update(headers)
        return request("GET", self._url(path, query), headers=hdrs, timeout=timeout)

    def post(
        self,
        path: str,
        json_body: dict | None = None,
        headers: dict | None = None,
        timeout: int = 10,
    ) -> tuple[Any, int, Dict[str, str]]:
        hdrs = dict(self.auth_headers)
        if headers:
            hdrs.update(headers)
        return request("POST", self._url(path), json_body=json_body, headers=hdrs, timeout=timeout)

    def put(
        self,
        path: str,
        json_body: dict | None = None,
        headers: dict | None = None,
        timeout: int = 10,
    ) -> tuple[Any, int, Dict[str, str]]:
        hdrs = dict(self.auth_headers)
        if headers:
            hdrs.update(headers)
        return request("PUT", self._url(path), json_body=json_body, headers=hdrs, timeout=timeout)

    def delete(
        self,
        path: str,
        headers: dict | None = None,
        timeout: int = 10,
    ) -> tuple[Any, int, Dict[str, str]]:
        hdrs = dict(self.auth_headers)
        if headers:
            hdrs.update(headers)
        return request("DELETE", self._url(path), headers=hdrs, timeout=timeout)

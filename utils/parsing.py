import json
from typing import Any, Dict, List


def parse_json_list(text: str) -> List[Any]:
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list")
    return data


def parse_json_object(text: str) -> Dict[str, Any]:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")
    return data

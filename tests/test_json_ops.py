import pytest
from pipeline.json_ops import extract_json, fix_truncated_json

def test_extract_json_valid():
    text = """Here is the JSON:
    ```json
    {"name": "test", "value": 123}
    ```
    Done."""
    result = extract_json(text)
    assert result == {"name": "test", "value": 123}

def test_extract_json_no_markdown():
    text = """{"key": "value"}"""
    result = extract_json(text)
    assert result == {"key": "value"}

def test_extract_json_multiple_objects_returns_first():
    text = """{"obj1": 1}{"obj2": 2}"""
    result = extract_json(text)
    assert result == {"obj1": 1}

def test_fix_truncated_json():
    # Missing final brackets
    truncated = '{"nodes": [{"id": "1"'
    fixed = fix_truncated_json(truncated)
    assert fixed.endswith('}]}')

    import json
    data = json.loads(fixed)
    assert data["nodes"][0]["id"] == "1"

def test_extract_json_auto_fixes_truncated():
    text = """Here is it:
    {"name": "test", "nodes": [{"id": "a"}"""
    result = extract_json(text)
    assert result["name"] == "test"
    assert result["nodes"][0]["id"] == "a"

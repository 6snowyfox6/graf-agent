from fastapi.testclient import TestClient
import pytest
from unittest.mock import patch, MagicMock

# Import the FastAPI app
from openai_api import app

client = TestClient(app)

def test_models_endpoint():
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert any(m["id"] == "graf-agent-pipeline" for m in data["data"])

@patch('openai_api.subprocess.run')
def test_chat_completions_prompt_too_long(mock_run):
    long_prompt = "A" * 17000
    payload = {
        "model": "graf-agent-general",
        "messages": [{"role": "user", "content": long_prompt}]
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 400
    assert "Prompt is too long" in response.json()["detail"]
    mock_run.assert_not_called()

@patch('openai_api.subprocess.run')
def test_chat_completions_missing_prompt(mock_run):
    payload = {
        "model": "graf-agent-general",
        "messages": []
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 400
    assert "User message not found" in response.json()["detail"]
    mock_run.assert_not_called()

@patch('openai_api.subprocess.run')
def test_chat_completions_success(mock_run):
    # Mock subprocess returning successfully
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_run.return_value = mock_proc

    payload = {
        "model": "graf-agent-general",
        "messages": [{"role": "user", "content": "Draw me a test architecture"}]
    }

    # Also mock _read_run_artifacts to return some fake data so it doesn't fail looking for missing files
    with patch('openai_api._read_run_artifacts') as mock_read:
        mock_read.return_value = (
            "...fake markdown...",
            "data:image/png;base64,fake",
            "```json\n{}\n```"
        )
        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) == 1
    content = data["choices"][0]["message"]["content"]
    assert "```json" in content
    assert "![Diagram]" in content

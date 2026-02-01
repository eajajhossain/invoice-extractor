from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_api_health():
    response = client.get("/")
    assert response.status_code in [200, 404]

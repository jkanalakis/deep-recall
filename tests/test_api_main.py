import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.middleware.auth import create_access_token


@pytest.fixture
def test_token():
    """Create a test JWT token."""
    return create_access_token(user_id="test_user", scopes=["read:memory", "write:memory", "delete:memory"])


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def test_api_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "timestamp" in response.json()


def test_api_root(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "deep-recall API Gateway"
    assert data["version"] == "0.1.0"
    assert "documentation" in data
    assert "health" in data


@pytest.mark.asyncio
async def test_api_memory_endpoints(client, test_token):
    """Test the memory-related endpoints."""
    # Test adding text to memory
    test_text = "This is a test text for the API"
    test_metadata = {"source": "api_test"}
    
    response = client.post(
        "/api/memory/add",
        json={"text": test_text, "metadata": test_metadata},
        headers={"Authorization": f"Bearer {test_token}"}  # Use the test token
    )
    assert response.status_code == 201  # Changed to 201 since that's what the endpoint returns
    assert "id" in response.json()  # Changed from memory_id to id
    
    # Test searching memory
    response = client.post(
        "/api/memory/query",
        json={"query": "test text API", "k": 1},
        headers={"Authorization": f"Bearer {test_token}"}  # Use the test token
    )
    assert response.status_code == 200
    response_data = response.json()
    assert isinstance(response_data, list)  # Check that it's a list
    assert len(response_data) > 0  # Check that we got at least one result
    assert "id" in response_data[0]  # Check that the first result has an ID
    assert "text" in response_data[0]  # Check that the first result has text 
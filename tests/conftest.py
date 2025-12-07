"""Pytest configuration and fixtures for API tests."""

import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Set test environment before importing app
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["S3_BUCKET"] = "test-bucket"

from src.api.main import app
from src.api.db.database import Base, get_db


# Create test database engine
TEST_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for tests."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database session for each test."""
    # Create all tables
    Base.metadata.create_all(bind=engine)

    session = TestingSessionLocal()
    yield session
    session.close()

    # Drop all tables after test
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_session):
    """Create a test client with database dependency override."""
    # Override database dependency
    app.dependency_overrides[get_db] = lambda: db_session

    # Create all tables
    Base.metadata.create_all(bind=engine)

    with TestClient(app) as test_client:
        yield test_client

    # Clean up
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def sample_artifact_data():
    """Sample artifact data for testing."""
    return {
        "name": "test-model",
        "url": "https://huggingface.co/test/model",
    }


@pytest.fixture
def sample_hf_url():
    """Sample HuggingFace URL for testing."""
    return "https://huggingface.co/google/gemma-3-270m"


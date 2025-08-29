"""Tests for LinkedIn Learning API client with retry logic."""

from unittest.mock import Mock, patch

import pytest
import requests
from requests import HTTPError, Response

from lil_api import LinkedInLearningAPIClient, LinkedInLearningAPIError


class TestLinkedInLearningAPIClient:
    """Test cases for LinkedIn Learning API client."""

    def test_init_default_values(self):
        """Test client initialization with default values."""
        client = LinkedInLearningAPIClient()

        assert client.base_url == "https://api.linkedin.com/v2"
        assert client.access_token is None
        assert client.max_retries == 5
        assert client.retry_delay == 1.0
        assert client.timeout == 30.0
        assert client.session is not None

    def test_init_with_access_token(self):
        """Test client initialization with access token."""
        token = "test_token"
        client = LinkedInLearningAPIClient(access_token=token)

        assert client.access_token == token
        assert "Authorization" in client.session.headers
        assert client.session.headers["Authorization"] == f"Bearer {token}"
        assert client.session.headers["X-Restli-Protocol-Version"] == "2.0.0"

    def test_init_custom_values(self):
        """Test client initialization with custom values."""
        client = LinkedInLearningAPIClient(
            base_url="https://custom.api.com",
            max_retries=3,
            retry_delay=0.5,
            timeout=60.0,
        )

        assert client.base_url == "https://custom.api.com"
        assert client.max_retries == 3
        assert client.retry_delay == 0.5
        assert client.timeout == 60.0

    @patch("lil_api.time.sleep")
    @patch("requests.Session.get")
    def test_get_success_first_attempt(self, mock_get, mock_sleep):
        """Test successful GET request on first attempt."""
        # Setup mock response
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        client = LinkedInLearningAPIClient()
        response = client.get("test-endpoint")

        # Verify request was made correctly
        mock_get.assert_called_once_with(
            "https://api.linkedin.com/v2/test-endpoint", params=None, timeout=30.0
        )
        assert response == mock_response
        mock_sleep.assert_not_called()

    @patch("lil_api.time.sleep")
    @patch("requests.Session.get")
    def test_get_429_with_retry_after_header(self, mock_get, mock_sleep):
        """Test 429 error handling with Retry-After header."""
        # Setup mock responses - first 429, then success
        mock_response_429 = Mock(spec=Response)
        mock_response_429.status_code = 429
        mock_response_429.headers = {"Retry-After": "30"}

        mock_response_success = Mock(spec=Response)
        mock_response_success.status_code = 200
        mock_response_success.raise_for_status.return_value = None

        mock_get.side_effect = [mock_response_429, mock_response_success]

        client = LinkedInLearningAPIClient()
        response = client.get("test-endpoint")

        # Verify retry logic
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once_with(30.0)  # Should use Retry-After value
        assert response == mock_response_success

    @patch("lil_api.time.sleep")
    @patch("requests.Session.get")
    def test_get_429_without_retry_after_header(self, mock_get, mock_sleep):
        """Test 429 error handling without Retry-After header (defaults to 60s)."""
        # Setup mock responses - first 429, then success
        mock_response_429 = Mock(spec=Response)
        mock_response_429.status_code = 429
        mock_response_429.headers = {}  # No Retry-After header

        mock_response_success = Mock(spec=Response)
        mock_response_success.status_code = 200
        mock_response_success.raise_for_status.return_value = None

        mock_get.side_effect = [mock_response_429, mock_response_success]

        client = LinkedInLearningAPIClient()
        response = client.get("test-endpoint")

        # Verify retry logic uses default 60 seconds
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once_with(60.0)  # Should use default 60s
        assert response == mock_response_success

    @patch("lil_api.time.sleep")
    @patch("requests.Session.get")
    def test_get_429_invalid_retry_after_header(self, mock_get, mock_sleep):
        """Test 429 error handling with invalid Retry-After header."""
        mock_response_429 = Mock(spec=Response)
        mock_response_429.status_code = 429
        mock_response_429.headers = {"Retry-After": "invalid"}

        mock_response_success = Mock(spec=Response)
        mock_response_success.status_code = 200
        mock_response_success.raise_for_status.return_value = None

        mock_get.side_effect = [mock_response_429, mock_response_success]

        client = LinkedInLearningAPIClient()
        response = client.get("test-endpoint")

        # Should fall back to default 60 seconds
        mock_sleep.assert_called_once_with(60.0)
        assert response == mock_response_success

    @patch("lil_api.time.sleep")
    @patch("requests.Session.get")
    def test_get_429_max_retries_exceeded(self, mock_get, mock_sleep):
        """Test 429 error when max retries are exceeded."""
        # Setup mock to always return 429
        mock_response_429 = Mock(spec=Response)
        mock_response_429.status_code = 429
        mock_response_429.headers = {"Retry-After": "1"}
        mock_get.return_value = mock_response_429

        client = LinkedInLearningAPIClient(max_retries=2)

        with pytest.raises(LinkedInLearningAPIError, match="Rate limit exceeded after 3 attempts"):
            client.get("test-endpoint")

        # Verify correct number of attempts and sleeps
        assert mock_get.call_count == 3  # max_retries + 1
        assert mock_sleep.call_count == 2  # max_retries

    @patch("lil_api.time.sleep")
    @patch("requests.Session.get")
    def test_get_exponential_backoff(self, mock_get, mock_sleep):
        """Test exponential backoff for multiple 429 errors."""
        # Setup multiple 429 responses then success
        mock_response_429 = Mock(spec=Response)
        mock_response_429.status_code = 429
        mock_response_429.headers = {}  # No Retry-After header

        mock_response_success = Mock(spec=Response)
        mock_response_success.status_code = 200
        mock_response_success.raise_for_status.return_value = None

        mock_get.side_effect = [
            mock_response_429,  # First attempt
            mock_response_429,  # Second attempt
            mock_response_success,  # Third attempt
        ]

        client = LinkedInLearningAPIClient(retry_delay=1.0)
        response = client.get("test-endpoint")

        # Verify exponential backoff (but rate limiting uses 60s minimum)
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2
        # Both should be 60.0 since it's rate limiting
        mock_sleep.assert_any_call(60.0)  # First retry
        mock_sleep.assert_any_call(60.0)  # Second retry
        assert response == mock_response_success

    @patch("requests.Session.get")
    def test_get_http_error_non_429(self, mock_get):
        """Test handling of HTTP errors other than 429."""
        mock_response = Mock(spec=Response)
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        client = LinkedInLearningAPIClient()

        with pytest.raises(LinkedInLearningAPIError, match="Non-retryable error"):
            client.get("test-endpoint")

    @patch("lil_api.time.sleep")
    @patch("requests.Session.get")
    def test_get_connection_error_with_retry(self, mock_get, mock_sleep):
        """Test connection error handling with retry."""
        # First call raises ConnectionError, second succeeds
        mock_response_success = Mock(spec=Response)
        mock_response_success.status_code = 200
        mock_response_success.raise_for_status.return_value = None

        mock_get.side_effect = [
            requests.exceptions.ConnectionError("Connection failed"),
            mock_response_success,
        ]

        client = LinkedInLearningAPIClient()
        response = client.get("test-endpoint")

        # Verify retry occurred
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once_with(1.0)  # First retry with base delay
        assert response == mock_response_success

    @patch("requests.Session.get")
    def test_get_non_retryable_error(self, mock_get):
        """Test handling of non-retryable errors."""
        # Simulate a 401 Unauthorized error (should not retry)
        mock_response = Mock(spec=Response)
        mock_response.status_code = 401
        http_error = HTTPError("401 Unauthorized")
        http_error.response = mock_response
        mock_get.side_effect = http_error

        client = LinkedInLearningAPIClient()

        with pytest.raises(LinkedInLearningAPIError, match="Non-retryable error"):
            client.get("test-endpoint")

        # Should only be called once (no retries)
        assert mock_get.call_count == 1

    def test_get_retry_delay_with_retry_after(self):
        """Test _get_retry_delay with Retry-After header."""
        client = LinkedInLearningAPIClient()

        mock_response = Mock()
        mock_response.headers = {"Retry-After": "45"}

        delay = client._get_retry_delay(mock_response, 0)
        assert delay == 45.0

    def test_get_retry_delay_without_retry_after(self):
        """Test _get_retry_delay without Retry-After header."""
        client = LinkedInLearningAPIClient(retry_delay=2.0)

        mock_response = Mock()
        mock_response.headers = {}

        # For rate limiting, should use max of exponential delay and 60s
        delay = client._get_retry_delay(mock_response, 0)
        assert delay == 60.0  # Max of (2.0 * 2^0 = 2.0) and 60.0

        delay = client._get_retry_delay(mock_response, 6)
        assert delay == 128.0  # Max of (2.0 * 2^6 = 128.0) and 60.0

    def test_is_retryable_error(self):
        """Test _is_retryable_error method."""
        client = LinkedInLearningAPIClient()

        # Connection errors should be retryable
        assert client._is_retryable_error(requests.exceptions.ConnectionError())
        assert client._is_retryable_error(requests.exceptions.Timeout())

        # 5xx HTTP errors should be retryable
        mock_response_500 = Mock()
        mock_response_500.status_code = 500
        http_error_500 = HTTPError()
        http_error_500.response = mock_response_500
        assert client._is_retryable_error(http_error_500)

        # 429 should be retryable
        mock_response_429 = Mock()
        mock_response_429.status_code = 429
        http_error_429 = HTTPError()
        http_error_429.response = mock_response_429
        assert client._is_retryable_error(http_error_429)

        # 4xx errors (except 429) should not be retryable
        mock_response_401 = Mock()
        mock_response_401.status_code = 401
        http_error_401 = HTTPError()
        http_error_401.response = mock_response_401
        assert not client._is_retryable_error(http_error_401)

    def test_context_manager(self):
        """Test context manager functionality."""
        with patch.object(LinkedInLearningAPIClient, "close") as mock_close:
            with LinkedInLearningAPIClient() as client:
                assert isinstance(client, LinkedInLearningAPIClient)
            mock_close.assert_called_once()

    def test_close(self):
        """Test close method."""
        client = LinkedInLearningAPIClient()
        with patch.object(client.session, "close") as mock_session_close:
            client.close()
            mock_session_close.assert_called_once()

    def test_base_url_normalization(self):
        """Test that base URL trailing slashes are handled correctly."""
        client = LinkedInLearningAPIClient(base_url="https://api.example.com/")
        assert client.base_url == "https://api.example.com"

        # Test URL construction
        with patch.object(client.session, "get") as mock_get:
            mock_response = Mock(spec=Response)
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            client.get("/test-endpoint")
            mock_get.assert_called_once_with(
                "https://api.example.com/test-endpoint", params=None, timeout=30.0
            )

    def test_get_with_params(self):
        """Test GET request with query parameters."""
        client = LinkedInLearningAPIClient()

        with patch.object(client.session, "get") as mock_get:
            mock_response = Mock(spec=Response)
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            params = {"count": 10, "start": 0}
            client.get("test-endpoint", params=params)

            mock_get.assert_called_once_with(
                "https://api.linkedin.com/v2/test-endpoint", params=params, timeout=30.0
            )

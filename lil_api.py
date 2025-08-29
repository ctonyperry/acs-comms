"""LinkedIn Learning API client with retry logic for rate limiting."""

import logging
import time
from typing import Any

import requests
from requests import Response

logger = logging.getLogger(__name__)


class LinkedInLearningAPIError(Exception):
    """Exception raised when LinkedIn Learning API operations fail."""

    pass


class LinkedInLearningAPIClient:
    """LinkedIn Learning API client with exponential backoff for rate limiting.

    Handles HTTP 429 (Too Many Requests) errors with exponential backoff and retry logic.
    """

    def __init__(
        self,
        base_url: str = "https://api.linkedin.com/v2",
        access_token: str | None = None,
        max_retries: int = 5,
        retry_delay: float = 1.0,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the LinkedIn Learning API client.

        Args:
            base_url: Base URL for the LinkedIn Learning API
            access_token: OAuth access token for authentication
            max_retries: Maximum number of retry attempts for rate limiting
            retry_delay: Base delay in seconds for exponential backoff
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.access_token = access_token
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

        # Configure session with default headers
        self.session = requests.Session()
        if access_token:
            self.session.headers.update(
                {
                    "Authorization": f"Bearer {access_token}",
                    "X-Restli-Protocol-Version": "2.0.0",
                }
            )

    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> Response:
        """Make a GET request with retry logic for rate limiting.

        Args:
            endpoint: API endpoint path (without base URL)
            params: Query parameters for the request

        Returns:
            requests.Response object

        Raises:
            LinkedInLearningAPIError: When all retry attempts are exhausted
            HTTPError: For non-rate-limit HTTP errors
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Making GET request to {url} (attempt {attempt + 1})")
                response = self.session.get(url, params=params, timeout=self.timeout)

                # Check for rate limiting
                if response.status_code == 429:
                    if attempt == self.max_retries:
                        logger.error(f"Rate limit exceeded after {self.max_retries + 1} attempts")
                        raise LinkedInLearningAPIError(
                            f"Rate limit exceeded after {self.max_retries + 1} attempts"
                        )

                    # Handle rate limiting with exponential backoff
                    retry_after = self._get_retry_delay(response, attempt)
                    logger.warning(
                        f"Rate limit hit (429) on attempt {attempt + 1}. "
                        f"Retrying in {retry_after:.1f} seconds..."
                    )
                    time.sleep(retry_after)
                    continue

                # Raise for other HTTP errors
                response.raise_for_status()
                logger.debug(f"Request successful: {response.status_code}")
                return response

            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries:
                    logger.error(f"Request failed after {self.max_retries + 1} attempts: {e}")
                    raise LinkedInLearningAPIError(
                        f"Request failed after {self.max_retries + 1} attempts: {e}"
                    ) from e

                # Only retry on certain exceptions (not auth errors, etc.)
                if self._is_retryable_error(e):
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(
                        f"Request failed on attempt {attempt + 1}: {e}. "
                        f"Retrying in {delay:.1f} seconds..."
                    )
                    time.sleep(delay)
                else:
                    # Don't retry for non-retryable errors
                    raise LinkedInLearningAPIError(f"Non-retryable error: {e}") from e

        # This should never be reached due to the loop structure
        raise LinkedInLearningAPIError("Unexpected error in retry logic")

    def _get_retry_delay(self, response: Response, attempt: int) -> float:
        """Calculate retry delay based on Retry-After header or exponential backoff.

        Args:
            response: HTTP response with 429 status
            attempt: Current attempt number (0-based)

        Returns:
            Delay in seconds before next retry
        """
        # Check for Retry-After header
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                # Retry-After can be in seconds (integer) or HTTP-date format
                # For simplicity, we'll assume it's in seconds
                delay = float(retry_after)
                logger.info(f"Using Retry-After header value: {delay} seconds")
                return delay
            except (ValueError, TypeError):
                logger.warning(f"Invalid Retry-After header value: {retry_after}")

        # Default to 60 seconds as specified in the requirements
        # or use exponential backoff if smaller
        exponential_delay = self.retry_delay * (2**attempt)
        default_delay = 60.0

        # Use the larger of exponential backoff or 60 seconds for rate limiting
        delay = max(exponential_delay, default_delay)
        logger.info(f"Using calculated delay: {delay} seconds")
        return float(delay)

    def _is_retryable_error(self, error: requests.exceptions.RequestException) -> bool:
        """Determine if an error is retryable.

        Args:
            error: The request exception that occurred

        Returns:
            True if the error should be retried, False otherwise
        """
        # Retry on connection errors, timeouts, and server errors (5xx)
        if isinstance(
            error,
            requests.exceptions.ConnectionError | requests.exceptions.Timeout,
        ):
            return True

        # Check for HTTP errors
        if isinstance(error, requests.exceptions.HTTPError):
            # Retry on server errors (5xx) but not client errors (4xx) except 429
            status_code = error.response.status_code if error.response else None
            return status_code is not None and (status_code >= 500 or status_code == 429)

        return False

    def close(self) -> None:
        """Close the HTTP session."""
        if self.session:
            self.session.close()

    def __enter__(self) -> "LinkedInLearningAPIClient":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Context manager exit."""
        self.close()

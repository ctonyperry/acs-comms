"""Main script demonstrating LinkedIn Learning API client usage."""

import logging
import os
import sys
from typing import Any

from lil_api import LinkedInLearningAPIClient, LinkedInLearningAPIError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Main function demonstrating API client usage."""
    # Get access token from environment variable
    access_token = os.getenv("LINKEDIN_ACCESS_TOKEN")
    if not access_token:
        logger.warning("No LINKEDIN_ACCESS_TOKEN environment variable found")
        logger.info("Running without authentication (will likely get 401 errors)")

    # Initialize the API client
    client = LinkedInLearningAPIClient(
        access_token=access_token,
        max_retries=5,  # Default value, specified per requirements
        retry_delay=1.0,
    )

    try:
        # Example API call - get learning assets
        logger.info("Making API request to LinkedIn Learning...")

        # This is an example endpoint - in practice you'd use actual LinkedIn Learning endpoints
        endpoint = "learningAssets"
        params: dict[str, Any] = {
            "count": 10,
            "start": 0,
        }

        response = client.get(endpoint, params=params)

        # Process the response
        logger.info(f"API request successful! Status: {response.status_code}")
        logger.info(f"Response headers: {dict(response.headers)}")

        try:
            data = response.json()
            logger.info(
                f"Response data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}"
            )
        except Exception as e:
            logger.warning(f"Could not parse JSON response: {e}")
            logger.info(f"Raw response: {response.text[:200]}...")

    except LinkedInLearningAPIError as e:
        logger.error(f"LinkedIn Learning API error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        # Clean up
        client.close()
        logger.info("API client closed")


if __name__ == "__main__":
    main()

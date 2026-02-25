#!/usr/bin/env python3
"""
Test exception handling in shutdown process.
"""

import logging
from src.core.shutdown_handler import get_shutdown_handler, register_cleanup

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def failing_cleanup():
    """A cleanup function that raises an exception."""
    logger.info("This cleanup will fail...")
    raise RuntimeError("Intentional error in cleanup")


def successful_cleanup():
    """A cleanup function that succeeds."""
    logger.info("This cleanup will succeed")


def main():
    """Test exception handling."""
    logger.info("Testing exception handling in shutdown...")
    
    shutdown_handler = get_shutdown_handler()
    
    # Register cleanups - one will fail, one will succeed
    register_cleanup(successful_cleanup, name="successful_cleanup_1")
    register_cleanup(failing_cleanup, name="failing_cleanup")
    register_cleanup(successful_cleanup, name="successful_cleanup_2")
    
    # Trigger shutdown
    shutdown_handler.trigger_shutdown()
    
    logger.info("Shutdown completed despite exception in cleanup")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

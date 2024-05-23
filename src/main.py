import logging

from discyril.engine import Engine

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)

    logger.info("Starting Discyril...")

    engine = Engine()
    engine.bootstrap()
    engine.start()

    logger.info("Exited Discyril. Bye!")


if __name__ == "__main__":
    main()

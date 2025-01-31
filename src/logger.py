import logging

def get_logger() -> logging.Logger:
    """Returns a logger

    Returns:
        logging.Logger: _description_
    """
    logger = logging.getLogger('dataflow')
    logger.setLevel(logging.INFO)
    return logger
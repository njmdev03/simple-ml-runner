import logging

def get_logger(name=None):
    if name:
        return logging.getLogger(f"mltool.{name}")

    return logging.getLogger("mltool")

logger = get_logger()
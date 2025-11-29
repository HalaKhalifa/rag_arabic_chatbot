import logging

# create logger instance
logger = logging.getLogger("ragchat")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
)

# stream handler (print to console)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# avoid duplicate handlers
if not logger.handlers:
    logger.addHandler(console_handler)
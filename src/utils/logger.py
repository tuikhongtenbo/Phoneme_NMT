"""
Logger utility for NMT training with colorful console output and file logging.
"""

import atexit
import logging
from termcolor import colored
from datetime import datetime
import functools
import sys
import os
from pathlib import Path


class ColorfulFormatter(logging.Formatter):
    
    def __init__(self, *args, **kwargs):
        super(ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        created_time = datetime.fromtimestamp(record.created)
        asctime = created_time.strftime(self.datefmt)
        levelname = record.levelname
        message = record.message
        log = self._fmt % {"asctime": asctime, "levelname": levelname, "message": message}

        # Apply colors based on log level
        if record.levelno == logging.DEBUG:
            log = colored(log, "blue")
        elif record.levelno == logging.INFO:
            log = colored(log, "green")
        elif record.levelno == logging.WARNING:
            log = colored(log, "yellow")
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            log = colored(log, "red")
        
        return log


@functools.lru_cache()  # Cache to prevent multiple handlers
def setup_logger(output=None, distributed_rank=0, *, color=True, name="PhonemeNMT"):
    """
    Initialize the PhonemeNMT logger and set its verbosity level to "DEBUG".

    Args:
        output (str, optional): A file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        distributed_rank (int): Rank of the process in distributed training (0 for master, >0 for workers).
            Only master process logs to console.
        color (bool): Whether to use colors in console output.
        name (str): The root module name of this logger.

    Returns:
        logging.Logger: A configured logger instance.

    Example:
        >>> logger = setup_logger(output="logs/train.log", name="NMT")
        >>> logger.info("Starting training...")
        >>> logger.debug("Batch size: 32")
        >>> logger.warning("GPU memory usage is high")
        >>> logger.error("Failed to load checkpoint")
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    FORMAT = "[%(asctime)s] %(levelname)s: %(message)s"
    plain_formatter = logging.Formatter(FORMAT, datefmt="%d/%m/%Y %H:%M:%S")
    
    # stdout logging: master only
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = ColorfulFormatter(fmt=FORMAT, datefmt="%d/%m/%Y %H:%M:%S")
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        
        if distributed_rank > 0:
            filename = filename + f".rank{distributed_rank}"
        
        # Create directory if it doesn't exist
        dirname = os.path.dirname(filename)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger


@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    """
    Cache the opened file object, so that different calls to `setup_logger`
    with the same file name can safely write to the same file.
    
    Args:
        filename (str): Path to log file
        
    Returns:
        file: Opened file stream
    """
    # Use 1K buffer if writing to cloud storage, -1 for default otherwise
    io = open(filename, "a", buffering=1024 if "://" in filename else -1)
    atexit.register(io.close)
    return io


if __name__ == "__main__":

    print("=" * 60)
    print("Testing PhonemeNMT Logger")
    print("=" * 60)
    
    # Test 1: Console only 
    print("\n1. Testing console logger:")
    logger1 = setup_logger(name="TestLogger1")
    logger1.debug("This is a DEBUG message - detailed information")
    logger1.info("This is an INFO message - general information")
    logger1.warning("This is a WARNING message - something might be wrong")
    logger1.error("This is an ERROR message - something went wrong")
    logger1.critical("This is a CRITICAL message - serious error occurred")
    
    # Test 2: Console + File logging
    print("\n2. Testing logger with file output:")
    log_dir = Path("logs/test")
    logger2 = setup_logger(output=str(log_dir), name="TestLogger2")
    logger2.info("Training started at epoch 1")
    logger2.debug("Model configuration: embed_dim=512, hidden_dim=512")
    logger2.warning("GPU memory usage: 85%")
    logger2.info("Batch processed: 100/1000")
    logger2.error("Failed to save checkpoint")
    print(f"\nLog file saved to: {log_dir}/log.txt")
    
    # Test 3: Specific log file
    print("\n3. Testing logger with specific log file:")
    logger3 = setup_logger(output="logs/test_custom.log", name="TestLogger3")
    logger3.info("This message goes to logs/test_custom.log")
    logger3.debug("Custom log file test")
    print("\nLog file saved to: logs/test_custom.log")
    
    # Test 4: No color
    print("\n4. Testing logger without colors:")
    logger4 = setup_logger(color=False, name="TestLogger4")
    logger4.info("This is without colors")
    logger4.warning("Plain text logging")
    
    print("\n" + "=" * 60)
    print("All logger tests completed!")
    print("=" * 60)
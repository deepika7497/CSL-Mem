import logging
import os

def setup_logger(
        logger_name='Default Logger',
        logfile_name='default.log',
        logfile_path='./logs'):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)


    handler = logging.FileHandler(os.path.join(logfile_path, logfile_name))
    formatter = logging.Formatter(
        fmt=f'%(asctime)s %(levelname)-8s %(message)s ',
        datefmt='%Y-%m-%d %H:%M:%S')

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
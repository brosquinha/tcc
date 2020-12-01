import logging
import sys


class Loggable():

    def __init__(self, log_level='debug'):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(getattr(logging, log_level.upper()))
            handler.setFormatter(logging.Formatter('%(asctime)-15s %(name)20s %(levelname)7s %(message)s'))
            self.logger.addHandler(handler)

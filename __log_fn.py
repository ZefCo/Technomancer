import logging
import pathlib
cwd = pathlib.Path.cwd()

def setup_logs(log_file: pathlib.Path, level = logging.DEBUG):
    '''
    This initializes the logs to go to one file. This is the most convinent way to do the logs. It captures everything.
    '''

    logging.basicConfig(level = level,
                        format = "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s | %(message)s",
                        handlers = [logging.FileHandler(log_file)])


# I'm leaving this here in case I change my mind and want to consider multiple different log files going to several different areas. That would be great for
# avoiding the large amount of DEBUG logging that is done right now, but the DEBUG logging does have a few useful things, as it shows the IP and port of
# Gradio, meaning I don't need to try to find that (Gradio uses 7860 by default and assumes that if you change it you know what you're doing. The problem I've
# seen in the past is you can set something to be a port and then it doesn't actually do that because the network is blocking that port. Ultimatly the initialization
# of Gradio is more complex than "type in python Technomancer.py and go!"). But the logs are sent to a single file right now.

    # log_dir = cwd / "Logs"
    # log_dir.mkdir(exist_ok = True)
    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#     logger = init_logger(name, level)

#     formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
#     handler = logging.FileHandler(log_file)
#     handler.setFormatter(formatter)
    
#     logger.addHandler(handler)

#     return logger


# def init_logger(name: str, level = logging.DEBUG):
#     '''
#     '''
#     logger = logging.getLogger(name)
#     logger.setLevel(level)

#     return logger

import logging
import pathlib
from datetime import datetime
cwd = pathlib.Path.cwd()

def setup_logs(script_name: str):
    '''
    Makes sure the logs are setup and ready to go. This should be running before the other scripts are accessed.

    Log levels: [NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL]
    '''
    log_dir = cwd / "Logs"
    log_dir.mkdir(exist_ok = True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    logging.basicConfig(level = logging.DEBUG,
                        format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                        handlers = [logging.FileHandler(log_dir / f"{script_name}_{timestamp}.log")])

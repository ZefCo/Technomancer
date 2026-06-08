from os.path import basename
from __log_fn import setup_logs
import logging
import pathlib

import gradio as gr

# from __states import log_file_state
from __tech_fn import live_stream, change_state, update_textbox, update_textbox_label
cwd = pathlib.Path.cwd()

def create_log():
    '''
    Creates a tab to check if the logs.
    '''
    with gr.Blocks() as logs:
        log_fe = gr.FileExplorer(label = "List of Log Files. Most rescent ones will be at the bottom", root_dir = str(cwd / "Logs"), file_count = "single", glob = "*.log", height = 100)
        # log_dd = gr.Dropdown(choices = [], label = "Log Files", info = "Looks at log files. Will live stream those files.")
        log_box = gr.TextArea(label = "Log File Stream", interactive = False, lines = 100)

        log_fe.input(fn = update_textbox_label, inputs = [log_fe], outputs = [log_box]).then(fn = live_stream, inputs = [log_fe], outputs = [log_box])

    return logs, {"log_box": log_box}

if __name__ in "__main__":
    from datetime import datetime
    # import pathlib

    # cwd = pathlib.Path.cwd()    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    

    log_dir = cwd / "Logs"
    log_dir.mkdir(parents = True, exist_ok = True)
    
    logger = setup_logs(__name__, log_dir / f"{pathlib.Path(basename(__file__)).stem}_{timestamp}.log")

    TECH_LOG, log_components = create_log()
else:
    logger = logging.getLogger(__name__)
    print("Rendered Upload Tab")
    logger.info("Rendered Upload Tab @ (time to be implemented)")

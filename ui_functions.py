import pathlib
cwd = pathlib.Path.cwd()
import yaml



def load_settings():
    '''
    Imports the settings
    '''

    with open(cwd / "Settings" / "UserOptions.yaml", "r") as file:
        settings = yaml.safe_load(file)

    return settings


def write_rules(settings: dict, updates, index):
    '''
    Rewrites the setting file for the user.
    '''
    settings[index] = updates
    with open(cwd / "Settings" / "UserOptions.yaml", "w") as file:
        yaml.dump(settings, file)
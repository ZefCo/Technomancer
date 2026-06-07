import pathlib
from os.path import basename
cwd = pathlib.Path.cwd()
test = basename(__file__)
test = pathlib.Path(test).stem

print(test)
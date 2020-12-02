import subprocess
from subprocess import Popen, PIPE
import sys

p = Popen(["python", "core.py"],
          stdin=PIPE,
          universal_newlines=True)

p.stdin.close()

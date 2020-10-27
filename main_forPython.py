import subprocess
from subprocess import Popen, PIPE
import sys

p = Popen(["python", "test.py"],
          stdin=PIPE,
          universal_newlines=True)

p.stdin.write("5\n")
p.stdin.write("5\n")
p.stdin.close()

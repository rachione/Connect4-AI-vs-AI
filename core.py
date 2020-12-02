import os
import signal
import sys
import pexpect
import time
from pexpect import popen_spawn


class Connect4:

    def __init__(self):
        self.proc = popen_spawn.PopenSpawn(
            'GAME230-P1-Connect_Four.exe', logfile=sys.stdout.buffer)

    def read(self):
        self.proc.expect('.*: ')

    def send(self, arg):
        self.proc.sendline(arg)

    def readAndSend(self, arg):
        self.read()
        self.send(arg)

    def start(self):
        args = ['4', '4', '4', '2', '2', '1']
        for arg in args:
            self.readAndSend(arg)
        self.playChess()
        self.kill()

    def playChess(self):
        self.readAndSend('1')

    def kill(self):
        os.kill(self.proc.pid, signal.SIGTERM)


if __name__ == "__main__":
    c4 = Connect4()
    c4.start()

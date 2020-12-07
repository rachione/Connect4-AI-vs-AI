import os
import signal
import sys
import pexpect
import time
import neat
from pexpect import popen_spawn


class Connect4:
    def __init__(self):
        self.proc = popen_spawn.PopenSpawn('GAME230-P1-Connect_Four.exe',
                                           logfile=sys.stdout.buffer)

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
        self.read()
        self.playChess(1)
        self.playChess(2)
        self.kill()

    def playChess(self, number):
        self.send(str(number))
        self.read()

    def kill(self):
        os.kill(self.proc.pid, signal.SIGTERM)


def run_car(genomes, config):
    c4 = Connect4()
    c4.start()


if __name__ == "__main__":
    config_path = "./config-feedforward.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    p.run(run_c4, 1000)

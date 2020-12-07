import os
import signal
import sys
import time
import neat
import pexpect
from pexpect import popen_spawn


class Connect4:

    def __init__(self):
        self.proc = popen_spawn.PopenSpawn('GAME230-P1-Connect_Four.exe',
                                           logfile=sys.stdout.buffer)

    def read(self):
        self.proc.expect(': ')
        # byte to str
        return self.proc.before.decode("utf-8")

    def send(self, arg):
        self.proc.sendline(arg)

    def readAndSend(self, arg):
        self.read()
        self.send(arg)

    def isWon(self, text):
        youWon = 'Player O has won the game!'
        return youWon in text

    def isError(self, text):
        fullCol = 'That column is full. Please try a different column'
        youLost = 'Player X has won the game!'
        return fullCol in text or youLost in text

    def start(self, genome):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0

        args = ['4', '4', '4', '2', '2', '1']
        for arg in args:
            self.readAndSend(arg)
        self.read()

        while True:
            input = self.get_input()
            outputs = net.activate(input)
            answer = self.get_answer(outputs)
            text = self.playChess(answer)

            if self.isError(text):
                break
            elif self.isWon(text):
                genome.fitness += 100
                break
            genome.fitness += 1

        self.kill()

    def playChess(self, number):
        self.send(str(number))
        text = self.read()
        return text

    # input data
    def get_input(self):
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # output data
    def get_answer(self, outputs):
        return outputs.index(max(outputs)) + 1

    def kill(self):
        os.kill(self.proc.pid, signal.SIGTERM)


def run_c4(genomes, config):
    for _, g in genomes:
        c4 = Connect4()
        c4.start(g)


if __name__ == "__main__":
    config_path = "./config-feedforward.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    p.run(run_c4, 1000)

import os
import pygame
import signal
import sys
import re
import time
import neat
import pexpect
import math
from pexpect import popen_spawn
from enum import Enum, auto

Generation = 0


class Result(Enum):
    Full = auto()
    Won = auto()
    Tie = auto()
    Lost = auto()
    Other = auto()


class GameDebug:

    def __init__(self):
        self.screenW = 600
        self.screenH = 600
        pygame.init()
        pygame.display.set_caption('AI vs AI Connect4')
        self.screen = pygame.display.set_mode((self.screenW, self.screenH))
        self.data = []
        self.draw()

    def input(self, data):
        self.data = data
        self.draw()

    def drawGeneration(self):
        font = pygame.font.SysFont("Arial", 30)
        gtxt = font.render(
            'Generation: ' + str(Generation), True, (0, 0, 0))
        gtxt_rect = gtxt.get_rect()
        gtxtW, gtxtH = gtxt.get_size()
        gtxt_rect.center = (gtxtW / 2 + 10, gtxtH / 2 + 10)
        self.screen.blit(gtxt, gtxt_rect)
        pygame.display.flip()

    def drawResult(self, result):
        font = pygame.font.SysFont("Arial", 30)
        resultTxt = font.render(result.name, True, (0, 0, 0))
        resultTxt_rect = resultTxt.get_rect()
        resultTxt_rect.center = (self.screenW / 2, 150)
        self.screen.blit(resultTxt, resultTxt_rect)
        pygame.display.flip()

    def draw(self):
        data = self.data
        dataLen = len(data)
        self.screen.fill((255, 255, 255))
        self.drawGeneration()
        if dataLen == 0:
            return
        size = int(math.sqrt(dataLen))
        cellSize = 40
        chess = pygame.Surface((cellSize * size, cellSize * size))
        chess.fill((255, 255, 255))
        for i in range(size):
            for j in range(size):
                index = i * size + j
                if data[index] == 0:
                    continue
                pos = (cellSize / 2 + cellSize * j,
                       cellSize / 2 + cellSize * i)
                color = (0, 0, 255) if data[index] == 1 else (255, 0, 0)
                pygame.draw.circle(chess, color, pos, 15)

        chessW, chessH = chess.get_size()
        self.screen.blit(chess, ((self.screenW - chessW) /
                                 2, (self.screenH - chessH) / 2))
        pygame.display.flip()


class RemotedConnect4:

    def __init__(self):
        self.proc = popen_spawn.PopenSpawn('GAME230-P1-Connect_Four.exe')
        self.gameDebug = GameDebug()
        #,logfile=sys.stdout.buffer

    def read(self):
        self.proc.expect(': ')

    def send(self, arg):
        self.proc.sendline(arg)

    def readAndSend(self, arg):
        self.read()
        self.send(arg)

    def checkResult(self, text):
        isTie = 'Tie game!' in text
        isFull = 'That column is full. Please try a different column' in text
        isLost = 'Player X has won the game!' in text
        isWon = 'Player O has won the game!' in text
        if isFull:
            return Result.Full
        elif isLost:
            return Result.Lost
        elif isTie:
            return Result.Tie
        elif isWon:
            return Result.Won
        else:
            return Result.Other

    def start(self, genome):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0

        args = ['4', '4', '4', '2', '2', '1']
        for arg in args:
            self.readAndSend(arg)
        self.read()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.kill()
                    sys.exit(0)
            input = self.get_input()
            genome.fitness = self.get_reward(input)
            outputs = net.activate(input)
            answer = self.get_answer(outputs)
            self.playChess(answer)
            text = self.get_text()

            result = self.checkResult(text)
            if result == Result.Full:
                self.gameDebug.drawResult(result)
                break
            elif result == Result.Tie or result == Result.Lost or result == Result.Won:
                input = self.get_input()
                genome.fitness = self.get_reward(input)
                self.gameDebug.drawResult(result)
                break
            time.sleep(0.1)

        time.sleep(0.5)

        self.kill()

    def playChess(self, number):
        self.send(str(number))
        text = self.read()
        return text

    # calculate fitness
    def get_reward(self, input):
        return 1
    # get text from stdout

    def get_text(self):
        return self.proc.before.decode("utf-8").replace('\n', '').replace('\r', '')

    # input data
    def get_input(self):
        def getVal(c):
            if c == 'O':
                return 1
            elif c == 'X':
                return -1
            else:
                return 0

        text = self.get_text()
        match = re.findall('[1-9]([OX.]+)', text)

        data = match[len(match) - 1]
        data = [getVal(x) for x in data]
        self.gameDebug.input(data)

        return data

    # output data
    def get_answer(self, outputs):
        return outputs.index(max(outputs)) + 1

    def kill(self):
        os.kill(self.proc.pid, signal.SIGTERM)


def run_c4(genomes, config):
    global Generation
    Generation += 1

    for i in range(len(genomes)):
        c4 = RemotedConnect4()
        c4.start(genomes[i][1])


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

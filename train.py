import os
import pygame
import signal
import sys
import re
import time
import neat
import pexpect
import math
import itertools
import numpy as np
from pexpect import popen_spawn
from enum import Enum, auto

Generation = 0
ScreenW = 600
ScreenH = 600


class Chess(Enum):
    You = 1
    Ai = -1
    Empty = 0


class Result(Enum):
    ColFull = auto()
    Won = auto()
    Tie = auto()
    Lost = auto()
    Other = auto()


class GameDebug:

    def __init__(self, lineCount):
        self.lineCount = lineCount
        self.cellSize = 50
        self.chessSize = self.cellSize * self.lineCount
        self.screen = pygame.display.get_surface()
        self.screen.fill((255, 255, 255))
        self.chessSurface = pygame.Surface((self.chessSize, self.chessSize))
        self.drawChessInit()
        self.drawGeneration()

    def input(self, data):
        self.drawChess(data)

    def getFontRect(self, txt):
        font = pygame.font.SysFont("Arial", 30)
        return font.render(txt, True, (0, 0, 0)).get_rect()

    def drawTxt(self, txt, center):
        font = pygame.font.SysFont("Arial", 30)
        txtsurface = font.render(txt, True, (0, 0, 0))
        txtsurface_rect = txtsurface.get_rect()
        txtsurface_rect.center = center
        self.screen.blit(txtsurface, txtsurface_rect)
        pygame.display.update()

    def drawFitness(self, score):
        txt = 'Fitness: ' + str(score)
        rect = self.getFontRect(txt)
        self.drawTxt(txt, (rect.width / 2 + 10,
                           rect.height / 2 + 40))

    def drawGeneration(self):
        txt = 'Generation: ' + str(Generation)
        rect = self.getFontRect(txt)
        self.drawTxt(txt, (rect.width / 2 + 10,
                           rect.height / 2 + 10))

    def drawResult(self, result):
        txt = result.name
        self.drawTxt(txt, (ScreenW / 2, ScreenH - 50))

    def drawChessInit(self):
        # blue bg
        self.chessSurface.fill((37, 106, 229))
        self.drawChess([0] * self.lineCount * self.lineCount)

    def drawChess(self, data):
        def getColor(piece):
            if piece == 0:
                return (66, 66, 66)
            elif piece == 1:
                return (255, 237, 0)
            else:
                return (224, 32, 26)

        lineCount = self.lineCount
        cellSize = self.cellSize
        chessSize = self.chessSize
        chessSurface = self.chessSurface

        for i in range(lineCount):
            for j in range(lineCount):
                index = i * lineCount + j
                pos = (cellSize / 2 + cellSize * j,
                       cellSize / 2 + cellSize * i)
                color = getColor(data[index])
                pygame.draw.circle(chessSurface, color, pos, 18)

        self.screen.blit(chessSurface, ((ScreenW - chessSize) / 2,
                                        (ScreenH - chessSize) / 2))
        pygame.display.update()


class Evaluator:

    def __init__(self, chess, size, matrix):
        self.size = size
        self.chess = chess
        self.matrix = matrix
        self.score = {}
        for i in range(2, self.size + 1):
            self.score[i] = 0

    def evaluateLine(self, line):
        groups = [(x, len(list(y))) for x, y in itertools.groupby(line)]
        for key, count in groups:
            if (key == self.chess.value) & (count >= 2):
                self.score[count] += 1

    def evaluateMatrix(self):
        matrix = self.matrix
        size = self.size
        # check horizontally
        for i in range(size):
            self.evaluateLine(matrix[i])

        # check vertically
        for i in range(size):
            self.evaluateLine(matrix[:, i])

        # check obliquely
        for i in range(size - 1):
            x = [k for k in range(0, size - i)]
            y = [k for k in range(i, size)]
            self.evaluateLine(matrix[x, y])
            self.evaluateLine(matrix[x, [(size - k - 1) for k in y]])
            if i > 0:
                self.evaluateLine(matrix[y, x])
                self.evaluateLine(matrix[[(size - k - 1) for k in x], y])

    def getScore(self):
        totalScore = 0
        for key, value in self.score.items():
            if (key == 2):
                totalScore += 1 * value
            elif (key == 3):
                totalScore += 10 * value
            if (key == 4) & (value >= 1):
                totalScore += 100 * value

        return totalScore

    def evaluate(self):
        self.evaluateMatrix()
        return self.getScore()


class RemotedConnect4:

    def __init__(self):
        self.proc = popen_spawn.PopenSpawn('GAME230-P1-Connect_Four.exe')
        #,logfile=sys.stdout.buffer
        self.lineCount = 6
        self.gameDebug = GameDebug(self.lineCount)

    def read(self):
        self.proc.expect(': ')

    def send(self, arg):
        self.proc.sendline(arg)

    def readAndSend(self, arg):
        self.read()
        self.send(arg)

    def checkResult(self, text):
        isTie = 'Tie game!' in text
        isColFull = 'That column is full. Please try a different column' in text
        isLost = 'Player X has won the game!' in text
        isWon = 'Player O has won the game!' in text
        if isColFull:
            return Result.ColFull
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

        args = [str(self.lineCount), str(self.lineCount), '4', '2', '2', '1']
        for arg in args:
            self.readAndSend(arg)
        self.read()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.kill()
                    sys.exit(0)
            input = self.get_input()
            genome.fitness = self.evaluate_reward(input)
            outputs = net.activate(input)
            answer = self.get_answer(outputs)
            self.playChess(answer)
            text = self.get_text()

            result = self.checkResult(text)
            if result == Result.ColFull:
                genome.fitness = self.evaluate_reward(input)
                genome.fitness -= 1000
                self.gameDebug.drawFitness(genome.fitness)
                self.gameDebug.drawResult(result)
                break
            elif (result == Result.Tie) | (result == Result.Lost) | (
                    result == Result.Won):
                input = self.get_input()
                genome.fitness = self.evaluate_reward(input)
                self.gameDebug.drawFitness(genome.fitness)
                self.gameDebug.drawResult(result)
                break

        time.sleep(0.1)
        self.kill()

    def playChess(self, number):
        self.send(str(number))
        text = self.read()
        return text

    # calculate fitness
    def evaluate_reward(self, data):

        size = int(math.sqrt(len(data)))
        matrix = np.reshape(np.array(data), (size, size))

        yourScore = Evaluator(Chess.You, size, matrix).evaluate()
        AiScore = Evaluator(Chess.Ai, size, matrix).evaluate()

        return yourScore - AiScore

    # get text from stdout
    def get_text(self):
        return self.proc.before.decode("utf-8").replace('\n',
                                                        '').replace('\r', '')

    # input data
    def get_input(self):
        def getVal(c):
            chess = Chess.Empty
            if c == 'O':
                chess = Chess.You
            elif c == 'X':
                chess = Chess.Ai
            return chess.value

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
    pygame.init()
    pygame.display.set_caption('AI vs AI Connect4')
    pygame.display.set_mode((ScreenW, ScreenH))

    config_path = "./config-feedforward.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    p.run(run_c4, 1000)

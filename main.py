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

    def __init__(self):
        self.screenW = 600
        self.screenH = 600
        pygame.init()
        pygame.display.set_caption('AI vs AI Connect4')
        self.screen = pygame.display.set_mode((self.screenW, self.screenH))
        self.data = []

    def input(self, data):
        self.data = data
        self.draw()

    def drawFitness(self, score):
        font = pygame.font.SysFont("Arial", 30)
        gtxt = font.render('Fitness: ' + str(score), True, (0, 0, 0))
        gtxt_rect = gtxt.get_rect()
        gtxt_rect.center = (gtxt_rect.width / 2 + 10,
                            gtxt_rect.height / 2 + 40)
        self.screen.blit(gtxt, gtxt_rect)
        pygame.display.flip()

    def drawGeneration(self):
        font = pygame.font.SysFont("Arial", 30)
        gtxt = font.render('Generation: ' + str(Generation), True, (0, 0, 0))
        gtxt_rect = gtxt.get_rect()
        gtxt_rect.center = (gtxt_rect.width / 2 + 10,
                            gtxt_rect.height / 2 + 10)
        self.screen.blit(gtxt, gtxt_rect)
        pygame.display.flip()

    def drawResult(self, result):
        font = pygame.font.SysFont("Arial", 30)
        resultTxt = font.render(result.name, True, (0, 0, 0))
        resultTxt_rect = resultTxt.get_rect()
        resultTxt_rect.center = (self.screenW / 2, self.screenH - 50)
        self.screen.blit(resultTxt, resultTxt_rect)
        pygame.display.flip()

    def draw(self):
        def getColor(data):
            if data == 0:
                return (66, 66, 66)
            elif data == 1:
                return (255, 237, 0)
            else:
                return (224, 32, 26)

        data = self.data
        dataLen = len(data)
        cellSize = 50

        if dataLen == 0:
            return
        size = int(math.sqrt(dataLen))
        self.screen.fill((255, 255, 255))
        self.drawGeneration()
        chess = pygame.Surface((cellSize * size, cellSize * size))
        chess.fill((37, 106, 229))
        for i in range(size):
            for j in range(size):
                index = i * size + j
                pos = (cellSize / 2 + cellSize * j,
                       cellSize / 2 + cellSize * i)
                color = getColor(data[index])
                pygame.draw.circle(chess, color, pos, 18)

        chessW, chessH = chess.get_size()
        self.screen.blit(chess, ((self.screenW - chessW) / 2,
                                 (self.screenH - chessH) / 2))
        pygame.display.flip()


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

        args = ['6', '6', '4', '2', '2', '1']
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
            # time.sleep(0.1)

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
    config_path = "./config-feedforward.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    p.run(run_c4, 1000)

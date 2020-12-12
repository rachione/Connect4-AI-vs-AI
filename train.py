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
import pickle
import numpy as np
from pexpect import popen_spawn
from enum import Enum, auto

ConfigPath = "./config-feedforward.txt"
Generation = 0
ScreenW = 600
ScreenH = 600


# chess piece type
class Piece(Enum):
    You = 1
    Ai = -1
    Empty = 0


class Result(Enum):
    ColFull = auto()
    Win = auto()
    Draw = auto()
    Lose = auto()
    Nope = auto()


class GameWindow:
    def __init__(self, title):
        pygame.init()
        pygame.display.set_caption(title)
        pygame.display.set_mode((ScreenW, ScreenH))


class GameBase:
    def __init__(self, lineCount):
        self.lineCount = lineCount
        self.cellSize = 50
        self.chessSize = self.cellSize * self.lineCount
        self.screen = pygame.display.get_surface()
        self.screen.fill((255, 255, 255))
        self.chessSurface = pygame.Surface((self.chessSize, self.chessSize))
        self.drawChess([0] * self.lineCount * self.lineCount)

    def drawTxt(self, txt, center):
        font = pygame.font.SysFont("Arial", 30)
        txtsurface = font.render(txt, True, (0, 0, 0))
        txtsurface_rect = txtsurface.get_rect()
        txtsurface_rect.center = center
        self.screen.blit(txtsurface, txtsurface_rect)
        pygame.display.update()

    def drawResult(self, result):
        txt = result.name
        self.drawTxt(txt, (ScreenW / 2, ScreenH - 50))

    def drawChessForeground(self):
        lineCount = self.lineCount
        cellSize = self.cellSize
        chessSize = self.chessSize

        chessFG = pygame.Surface((self.chessSize, self.chessSize),
                                 pygame.SRCALPHA)
        chessFG.fill((37, 106, 229))
        for i in range(lineCount):
            for j in range(lineCount):
                pos = (cellSize / 2 + cellSize * j,
                       cellSize / 2 + cellSize * i)
                pygame.draw.circle(chessFG, (0, 0, 0, 0), pos, 18)

        self.screen.blit(chessFG, ((ScreenW - chessSize) / 2,
                                   (ScreenH - chessSize) / 2))

    def drawChessBackground(self):
        self.chessSurface.fill((66, 66, 66))

    def drawPiece(self, data):
        def getColor(piece):
            if piece == 1:
                return (255, 237, 0)
            else:
                return (224, 32, 26)

        lineCount = self.lineCount
        cellSize = self.cellSize
        chessSize = self.chessSize
        chessSurface = self.chessSurface

        for y in range(lineCount):
            for x in range(lineCount):
                index = y * lineCount + x
                piece = data[index]
                if piece == 0:
                    continue

                pos = (cellSize / 2 + cellSize * x,
                       cellSize / 2 + cellSize * y)
                color = getColor(piece)
                pygame.draw.circle(chessSurface, color, pos, 18)

        self.screen.blit(chessSurface, ((ScreenW - chessSize) / 2,
                                        (ScreenH - chessSize) / 2))

    def drawChess(self, data):
        self.drawChessBackground()
        self.drawPiece(data)
        self.drawChessForeground()
        pygame.display.update()


class GameDebug(GameBase):
    def __init__(self, lineCount):
        GameBase.__init__(self, lineCount)
        self.drawGeneration()

    def getFontRect(self, txt):
        font = pygame.font.SysFont("Arial", 30)
        return font.render(txt, True, (0, 0, 0)).get_rect()

    def drawFitness(self, score):
        txt = 'Fitness: ' + str(score)
        rect = self.getFontRect(txt)
        self.drawTxt(txt, (rect.width / 2 + 10, rect.height / 2 + 40))

    def drawGeneration(self):
        txt = 'Generation: ' + str(Generation)
        rect = self.getFontRect(txt)
        self.drawTxt(txt, (rect.width / 2 + 10, rect.height / 2 + 10))


class GameReplay(GameBase):
    def __init__(self, lineCount):
        GameBase.__init__(self, lineCount)


class Evaluator:
    def __init__(self, piece, size, matrix):
        self.size = size
        self.piece = piece
        self.matrix = matrix
        self.score = {}
        for i in range(2, self.size + 1):
            self.score[i] = 0

    def evaluateLine(self, line):
        groups = [(x, len(list(y))) for x, y in itertools.groupby(line)]
        for piece, count in groups:
            if (piece == self.piece.value) & (count >= 2):
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


class Connect4Commander:
    def __init__(self, genome, config, isDebug=True):
        self.genome = genome
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.proc = popen_spawn.PopenSpawn('GAME230-P1-Connect_Four.exe')
        self.lineCount = 6
        if isDebug:
            self.game = GameDebug(self.lineCount)
        else:
            self.game = GameReplay(self.lineCount)

    def read(self):
        self.proc.expect(': ')

    def send(self, arg):
        self.proc.sendline(arg)

    def readAndSend(self, arg):
        self.read()
        self.send(arg)

    def checkResult(self, text):
        isDraw = 'Draw game!' in text
        isColFull = 'That column is full. Please try a different column' in text
        isLose = 'Player X has won the game!' in text
        isWin = 'Player O has won the game!' in text
        if isColFull:
            return Result.ColFull
        elif isLose:
            return Result.Lose
        elif isDraw:
            return Result.Draw
        elif isWin:
            return Result.Win
        else:
            return Result.Nope

    def interactiveInit(self):
        args = [str(self.lineCount), str(self.lineCount), '4', '2', '2', '1']
        for arg in args:
            self.readAndSend(arg)
        self.read()

    def interact(self):
        # last chess Board
        inputNodeList = self.getChessBoards()
        outputNodes = self.net.activate(inputNodeList[-1])
        answer = self.get_answer(outputNodes)
        self.playChess(answer)
        text = self.get_text()
        result = self.checkResult(text)
        return inputNodeList, result

    def replay(self):
        self.interactiveInit()
        clock = pygame.time.Clock()
        result = Result.Nope
        while result == Result.Nope:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.kill()
                    sys.exit(0)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit(0)

            inputNodeList, result = self.interact()
            for inputNode in inputNodeList:
                self.game.drawChess(inputNode)
                time.sleep(0.2)
            if result != Result.Nope:
                if (result == Result.Draw) | (result == Result.Lose) | (
                        result == Result.Win):
                    inputNode = self.getChessBoards()[-1]
                    self.game.drawChess(inputNode)
                self.game.drawResult(result)

        time.sleep(0.5)
        self.kill()

    def train(self):
        self.interactiveInit()

        result = Result.Nope
        while result == Result.Nope:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.kill()
                    sys.exit(0)

            inputNodeList, result = self.interact()
            if result != Result.Nope:
                inputNode = inputNodeList[-1]
                if result == Result.ColFull:
                    self.genome.fitness = self.evaluateReward(inputNode)
                    self.genome.fitness -= 1000  # punish
                elif (result == Result.Draw) | (result == Result.Lose) | (
                        result == Result.Win):
                    inputNode = self.getChessBoards()[-1]
                    self.genome.fitness = self.evaluateReward(inputNode)

                self.game.drawChess(inputNode)
                self.game.drawFitness(self.genome.fitness)
                self.game.drawResult(result)

        time.sleep(0.3)
        self.kill()

    def playChess(self, number):
        self.send(str(number))
        text = self.read()
        return text

    # calculate fitness
    def evaluateReward(self, data):

        size = int(math.sqrt(len(data)))
        matrix = np.reshape(np.array(data), (size, size))

        yourScore = Evaluator(Piece.You, size, matrix).evaluate()
        AiScore = Evaluator(Piece.Ai, size, matrix).evaluate()

        return yourScore - AiScore

    # get text from stdout
    def get_text(self):
        return self.proc.before.decode("utf-8").replace('\n',
                                                        '').replace('\r', '')

    # get input node
    def getChessBoards(self):
        def getVal(c):
            chess = Piece.Empty
            if c == 'O':
                chess = Piece.You
            elif c == 'X':
                chess = Piece.Ai
            return chess.value

        text = self.get_text()
        matches = re.findall('[1-9]([OX.]+)', text)

        chessBoards = []
        for match in matches:
            data = [getVal(x) for x in match]
            chessBoards.append(data)

        return chessBoards

    # output data
    def get_answer(self, outputs):
        return outputs.index(max(outputs)) + 1

    def kill(self):
        os.kill(self.proc.pid, signal.SIGTERM)


class Trainer:
    def isTrained(self, genomes):
        bestElitism = max(genomes, key=lambda genome: genome[1].fitness)
        return bestElitism[1].fitness > 100

    def saveGenome(self, genomes):
        elitism = [g for g in genomes if g[1].fitness > 60]
        with open("elitism.pkl", "wb") as f:
            pickle.dump(elitism, f)

    def replayGenome(self):
        config = neat.config.Config(neat.DefaultGenome,
                                    neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet,
                                    neat.DefaultStagnation, ConfigPath)
        with open('elitism.pkl', "rb") as f:
            genomes = pickle.load(f)

        for i in range(len(genomes)):
            # no debug
            c4 = Connect4Commander(genomes[i][1], config, False)
            c4.replay()

    def run(self, genomes, config):
        global Generation
        Generation += 1

        for i in range(len(genomes)):
            c4 = Connect4Commander(genomes[i][1], config)
            c4.train()

        if self.isTrained(genomes):
            self.saveGenome(genomes)


if __name__ == "__main__":
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                ConfigPath)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    GameWindow('AI vs AI Connect4 Trainer')
    p.run(Trainer().run, 1000)

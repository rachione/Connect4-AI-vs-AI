import pygame
from train import Trainer, GameDebugWindow

if __name__ == '__main__':
    GameDebugWindow('AI vs AI Connect4')
    Trainer().replayGenome()
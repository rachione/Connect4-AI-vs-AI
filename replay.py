import pygame
from train import Trainer, GameWindow

if __name__ == '__main__':
    GameWindow('AI vs AI Connect4')
    Trainer().replayGenome()
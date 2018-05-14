from Othello import Othello
from Othello import AI
import numpy as np

generation = input('Which generation do you wish to play against:')
w1 = np.load('weight'+generation+'-1.npy')
w2 = np.load('weight'+generation+'-2.npy')
ai1 = AI(weight1=w1, weight2=w2)
w1 = np.load('weight'+str(int(generation)-1)+'-1.npy')
w2 = np.load('weight'+str(int(generation)-1)+'-2.npy')
ai2 = AI(weight1=w1, weight2=w2)
# player = int(input('Which player do you wish to be (black: -1, white: 1):'))
game = Othello()
# game.play(ai)
game.play_self(ai1, ai2,display_board=True)

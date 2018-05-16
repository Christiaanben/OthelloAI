from Othello import Othello
from Othello import AI
import numpy as np
import matplotlib.pyplot as plt

generation = int(input('Which generation do you wish to play against:'))
w1 = np.load('weights\weight-Gen{:03d}-1.npy'.format(generation))
w2 = np.load('weights\weight-Gen{:03d}-2.npy'.format(generation))
ai = AI(weight1=w1, weight2=w2)
player = int(input('Which player do you wish to be (black: -1, white: 1):'))
game = Othello()
game.play(ai, player)


# n, bins, patches = plt.hist(w1, 50, normed=1, facecolor='green', alpha=0.75)
# plt.show()
from Othello import Othello
from Othello import AI
import numpy as np

board_empty = [[0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]]

board_test = [[0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [1, -1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]]


ai = []
n_ai = 1000
score = np.zeros(n_ai, dtype='int')
for i in range(n_ai):
    ai.append(AI(name=i))
for i in range(n_ai):
    for j in range((i+1), n_ai):
        game = Othello()
        winner = game.play_self(ai[i], ai[j])
        score[winner.get_name()] += 1
final_winner = np.argmax(score)
print('AI-'+str(final_winner)+' won '+str(score[final_winner])+' out of '+str(n_ai-1)+' ('+str(score[final_winner]/(n_ai-1)*100)+'%) games.')
weight1, weight2, weight3 = ai[final_winner].get_weights()
np.save('weight1.npy', weight1)
np.save('weight2.npy', weight2)
np.save('weight3.npy', weight3)

# w1 = np.load('weight1.npy')
# w2 = np.load('weight2.npy')
# w3 = np.load('weight3.npy')
# ai = AI(player=1, weight1=w1, weight2=w2, weight3=w3)
# game = Othello()
# game.play(ai, -1)
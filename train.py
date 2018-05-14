from Othello import Othello
from Othello import AI
import numpy as np

n_ai = 10000
n_winners = 100
ai_winners = None

generation = 0
while True:
    generation += 1
    print('Generation '+str(generation)+':')
    ai = []
    score = np.zeros(n_ai, dtype='int')
    for i in range(n_ai):
        if ai_winners is not None:
            weight1, weight2 = ai_winners[int(i/(n_ai/n_winners))].mutate()
            ai.append(AI(name=i, weight1=weight1, weight2=weight2))
        else:
            ai.append((AI(name=i)))
    for i in range(n_ai):
        for j in range((i+1), n_ai):
            game = Othello()
            winner = game.play_self(ai[i], ai[j])
            score[winner.get_name()] += 1
    winners = score.argsort()[-n_winners:][::-1]
    ai_winners = []
    for winner in winners:
        ai_winners.append(ai[winner])
    final_winner = np.argmax(score)
    print('Ai-{} won {} out of {} ({:.2f}%) games.'.format(final_winner, score[final_winner], n_ai-1, score[final_winner]/(n_ai-1)*100))
    weight1, weight2 = ai[final_winner].get_weights()
    np.save('weight'+str(generation)+'-1.npy', weight1)
    np.save('weight'+str(generation)+'-2.npy', weight2)

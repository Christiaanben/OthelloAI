from Othello import Othello
from Othello import AI
import numpy as np

n_ai = 5000
n_winners = 50
restart = False
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
            if i % n_winners == 0:
                weight1, weight2 = ai_winners[int(i / (n_ai / n_winners))].get_weights()
            ai.append(AI(name=i, weight1=weight1, weight2=weight2))
        else:
            ai.append((AI(name=i)))
    prev_percentage = 0
    print('Complete: ', end='', flush=True)
    for i in range(n_ai):
        percentage = int(i/n_ai*100)
        [print('#', end='', flush=True) for x in range(percentage-prev_percentage)]
        prev_percentage = percentage
        for j in range((i+1), n_ai):
            game = Othello()
            winner = game.play_self(ai[i], ai[j])
            score[winner.get_name()] += 1
    print('| 100%')
    winners = score.argsort()[-n_winners:][::-1]
    ai_winners = []
    for winner in winners:
        ai_winners.append(ai[winner])
    final_winner = np.argmax(score)
    print('Ai-{} won {} out of {} ({:.2f}%) games.'.format(final_winner, score[final_winner], n_ai-1, score[final_winner]/(n_ai-1)*100))
    for i in range(len(ai_winners)):
        weight1, weight2 = ai_winners[i].get_weights()
        np.save('weights\weight-Gen{:02d}-{:02d}-1.npy'.format(generation, i), weight1)
        np.save('weights\weight-Gen{:02d}-{:02d}-2.npy'.format(generation, i), weight2)

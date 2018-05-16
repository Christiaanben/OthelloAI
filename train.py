from Othello import Othello
from Othello import AI
import numpy as np

n_ai = 151
n_winners = 50
# TODO implement restart function
restart = False
ai_winners = None
winner_of_generation = []
generation = 0
while True:
    generation += 1
    print('Generation '+str(generation)+':')
    ai = []
    for i in range(n_ai):
        if ai_winners is not None:
            weight1, weight2 = ai_winners[int(i/(n_ai/n_winners))].mutate(chance=0.02)
            if i % n_winners == 0:
                weight1, weight2 = ai_winners[int(i / (n_ai / n_winners))].get_weights()
            ai.append(AI(name=i, weight1=weight1, weight2=weight2))
        else:
            ai.append((AI(name=i)))
    prev_percentage = 0
    for i in range(len(winner_of_generation)):
        winner_of_generation[i].set_name(i+n_ai)
        ai.append(winner_of_generation[i])
    score = np.zeros(len(ai), dtype='int')
    print('Complete: ', end='', flush=True)
    for i in range(len(ai)):
        percentage = int(i/len(ai)*100)
        [print('#', end='', flush=True) for x in range(percentage-prev_percentage)]
        prev_percentage = percentage
        for j in range((i+1), len(ai)):
            game = Othello()
            winner = game.play_self(ai[i], ai[j])
            score[winner.get_name()] += 1
    print('| 100%')
    winners = score.argsort()[-n_winners:][::-1]
    ai_winners = []
    for winner in winners:
        ai_winners.append(ai[winner])
    print('Ai-{} won {} out of {} ({:.2f}%) games.'.format(winners[0], score[winners[0]], len(ai)-1, score[winners[0]]/(len(ai)-1)*100))

    # Adds the best AI of this generation to the list of contenders
    if ai_winners[0] not in winner_of_generation and (score[winners[0]]/(len(ai)-1)) > 0.65:
        winner_of_generation.append(ai_winners[0])

    # Store top player of each generation and the latest winners
    weight1, weight2 = ai_winners[0].get_weights()
    np.save('weights\weight-Gen{:03d}-1.npy'.format(generation), weight1)
    np.save('weights\weight-Gen{:03d}-2.npy'.format(generation), weight2)
    for i in range(len(ai_winners)):
        weight1, weight2 = ai_winners[i].get_weights()
        np.save('weights\weight-{:03d}-1.npy'.format(i), weight1)
        np.save('weights\weight-{:03d}-2.npy'.format(i), weight2)

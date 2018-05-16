from Othello import Othello
from Othello import AI
import numpy as np

n_ai = 1000
n_winners = 100
restart = True
ai_winners = None
winner_of_generation = []
generation = 0

if restart:
    file = open('weights\\best_gen.txt', 'w')
    file.close()
else:
    ai_winners = []
    for i in range(n_winners):
        w1 = np.load('weights\weight-{:03d}-1.npy'.format(i))
        w2 = np.load('weights\weight-{:03d}-2.npy'.format(i))
        ai_winners.append(AI(weight1=w1, weight2=w2))
        file = open('weights\\best_gen.txt', 'r')
    for line in file:
        w1 = np.load('weights\weight-Gen{:03d}-1.npy'.format(int(line)))
        w2 = np.load('weights\weight-Gen{:03d}-2.npy'.format(int(line)))
        winner_of_generation.append(AI(weight1=w1, weight2=w2))
        generation = int(line)
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
        file = open('weights\\best_gen.txt', 'a')
        file.write(str(generation)+'\n')
        file.close()

    # Store top player of each generation and the latest winners
    weight1, weight2 = ai_winners[0].get_weights()
    np.save('weights\weight-Gen{:03d}-1.npy'.format(generation), weight1)
    np.save('weights\weight-Gen{:03d}-2.npy'.format(generation), weight2)
    for i in range(len(ai_winners)):
        weight1, weight2 = ai_winners[i].get_weights()
        np.save('weights\weight-{:03d}-1.npy'.format(i), weight1)
        np.save('weights\weight-{:03d}-2.npy'.format(i), weight2)

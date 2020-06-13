
''' Test run the svpg package '''

import svpg

number_episodes = 401
number_of_runs = 1 #Repeat runs for reproducibility checks...

for k in range(1):

    #First, try the pure A2C
    svpg.train_svpg(number_of_episodes=number_episodes, pure_a2c=True, run = k)

    #Next, try SVPG with the conv layers updated
    svpg.train_svpg(number_of_episodes=number_episodes, pure_a2c=False, run = k)



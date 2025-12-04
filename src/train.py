import EasyGA as ga
import random
import numpy as np
import math
import time
from kesslergame import Scenario, KesslerGame, GraphicsType, TrainerEnvironment
from scott_dick_controller import ScottDickController
from test_controller_fuzzy import FuzzyController
from my_controller_v3 import MyController
from graphics_both import GraphicsBoth

# Define game scenario
my_test_scenario = Scenario(name='Test Scenario',
                            num_asteroids=10,
                            ship_states=[
                                {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 3},
                            ],
                            map_size=(1000, 800),
                            time_limit=60,
                            ammo_limit_multiplier=0,
                            stop_if_no_ammo=False)


# Define Game Settings
game_settings = {'perf_tracker': True,
                'graphics_type': GraphicsType.Tkinter,
                'realtime_multiplier': 1,
                'graphics_obj': None,
                'frequency': 30}

game = KesslerGame(settings=game_settings)  # Use this to visualize the game scenario
# game = TrainerEnvironment(settings=game_settings)  # Use this for max-speed, no-graphics simulation

def fitness(chromosome):
    pre = time.perf_counter()
    x = []
    for index_gene,gene in enumerate(chromosome.gene_list):
        x.append(gene.value)

    try:
        score, perf_data = game.run(scenario=my_test_scenario, controllers=[MyController(x)])
    except Exception as e:
        print(e)
        return 0

    # print('Scenario eval time: '+str(time.perf_counter()-pre))
    # print(score.stop_reason)
    # print('Asteroids hit: ' + str([team.asteroids_hit for team in score.teams]))
    # print('Deaths: ' + str([team.deaths for team in score.teams]))
    # print('Mean eval time: ' + str([team.mean_eval_time for team in score.teams]))
    # score.teams[0].accuracy
    print(score.teams[0].asteroids_hit*score.teams[0].accuracy)

    return score.teams[0].asteroids_hit*score.teams[0].accuracy

def chromosome_function():
    chromosome_data = np.sort([random.uniform(0, 3) for _ in range(6)]).tolist() # bullet_time
    chromosome_data.extend(np.sort([random.uniform(-math.pi,math.pi) for _ in range (10)]).tolist()) # theta_delta
    chromosome_data.extend(np.sort([random.uniform(1,4) for _ in range(5)]).tolist()) # asteroid size
    chromosome_data.extend(np.sort([random.uniform(-180,180) for _ in range(12)]).tolist()) # turn rate
    chromosome_data.extend(np.sort([random.uniform(0,1) for _ in range(3)]).tolist()) # fire command
    chromosome_data.extend(np.sort([random.uniform(0,1000) for _ in range(4)]).tolist()) # threat_distance
    chromosome_data.extend(np.sort([random.uniform(-500,500) for _ in range(5)])) # threat approach speed
    chromosome_data.extend(np.sort([random.uniform(0,500) for _ in range(4)])) # distance from map boundary
    chromosome_data.extend(np.sort([random.uniform(-500,500) for _ in range(6)])) # thrust (-ve = moving away, +ve = approaching))
    chromosome_data.extend(np.sort([random.uniform(0,1) for _ in range(5)])) # threat level (0-1)
    chromosome_data.extend(np.sort([random.uniform(0,1) for _ in range(3)])) # health status (0-1)
    chromosome_data.extend(np.sort([random.uniform(0,1) for _ in range(5)])) # survival override (0-1)
    return chromosome_data

if __name__ == "__main__":
    alg = ga.GA()
    alg.population_size = 50
    alg.generation_goal = 1
    alg.target_fitness_type = 'max'
    alg.fitness_function_impl = fitness
    alg.chromosome_impl = chromosome_function
    alg.evolve()

    alg.graph.highest_value_chromosome()
    alg.graph.show()







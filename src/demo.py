import EasyGA as ga
import random
import numpy as np
import sqlite3
import math
import os
import time
import ast
import re
from kesslergame import Scenario, KesslerGame, GraphicsType, TrainerEnvironment
from scott_dick_controller import ScottDickController
from my_controller_v3 import MyController
from graphics_both import GraphicsBoth

# Define game scenario
my_test_scenario = Scenario(name='Test Scenario',
                            num_asteroids=10,
                            ship_states=[
                                {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 3},
                                {'position': (400, 600), 'angle': 90, 'lives': 3, 'team': 2, "mines_remaining": 3},
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
    # pre = time.perf_counter()

    try:
        pre = time.perf_counter()
        score, perf_data = game.run(scenario=my_test_scenario, controllers=[MyController(chromosome), ScottDickController()])
    except Exception as e:
        print(e)
        return 0

    # print('Scenario eval time: '+str(time.perf_counter()-pre))
    # print(score.stop_reason)
    # print('Asteroids hit: ' + str([team.asteroids_hit for team in score.teams]))
    # print('Deaths: ' + str([team.deaths for team in score.teams]))
    # print('Mean eval time: ' + str([team.mean_eval_time for team in score.teams]))
    # score.teams[0].accuracy
    print("us " + str(score.teams[0].asteroids_hit))
    print("test "+ str(score.teams[1].asteroids_hit))

    return score.teams[0].asteroids_hit*score.teams[0].accuracy


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))   
    db_path = os.path.join(base_dir, "database4.db")        

    # print("Using DB at:", db_path)  # debug

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print("Tables in DB:", cursor.fetchall())


    cursor.execute("SELECT chromosome FROM data ORDER BY fitness DESC LIMIT 1;")
    records = cursor.fetchall()
    
    input_string_cleaned = re.sub(r'np\.float64\((.*?)\)', r'\1', records[0][0])
    output_list = ast.literal_eval(input_string_cleaned)

    # print(output_list)
    fitness(output_list)

    connection.close()
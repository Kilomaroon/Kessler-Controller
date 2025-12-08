import EasyGA as ga
import random
import numpy as np
import sqlite3
import math
import os
import time
import ast
import matplotlib.pyplot as plt
import re
from kesslergame import Scenario, KesslerGame, GraphicsType, TrainerEnvironment
from scott_dick_controller import ScottDickController
from my_controller_v3 import MyController
from graphics_both import GraphicsBoth

# Define game scenario
my_test_scenario = Scenario(name='Test Scenario',
                            num_asteroids=10,
                            ship_states=[
                                {'position': (400, 300), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 3},
                                {'position': (400, 500), 'angle': 90, 'lives': 3, 'team': 2, "mines_remaining": 3},
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

def fitness(chromosome):

    try:
        pre = time.perf_counter()
        score, perf_data = game.run(scenario=my_test_scenario, controllers=[MyController(chromosome), ScottDickController()])
    except Exception as e:
        print(e)
        return 0

    return score


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))   
    db_path = os.path.join(base_dir,"batches", "d0.db")        

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    cursor.execute("SELECT chromosome FROM data ORDER BY fitness DESC LIMIT 1;")
    records = cursor.fetchall()
    
    input_string_cleaned = re.sub(r'np\.float64\((.*?)\)', r'\1', records[0][0])
    output_list = ast.literal_eval(input_string_cleaned)

    # get user input for runs
    while True:
        user_input = input("Runs (int): ")
        try:
            runs = int(user_input)
            print(f"Starting {runs} runs...")
            break  # Exit the loop if conversion is successful
        except ValueError:
            print("Invalid input. Please enter a whole number.")

    # run and store data for plotting
    sdick_vals = {
        "accuracy": [],
        "score": []
    }
    student_vals = {
        "accuracy": [],
        "score": []
    }

    for i in range(runs):
        score = fitness(output_list)
        sdick_vals["accuracy"].append(score.teams[1].accuracy)
        student_vals["accuracy"].append(score.teams[0].accuracy)
        sdick_vals["score"].append(score.teams[1].asteroids_hit)
        student_vals["score"].append(score.teams[0].asteroids_hit)

    connection.close()

    plt.figure(1)
    plt.plot(range(runs), sdick_vals["score"], label='Dr Dick Scores', color='blue', marker='o')
    plt.plot(range(runs), student_vals["score"], label='Student Score', color='red', marker='x')

    plt.title("Score Comparasion")
    plt.legend()

    plt.figure(2)
    plt.plot(range(runs), sdick_vals["accuracy"], label='Dr Dick Accuracy', color='blue', marker='o')
    plt.plot(range(runs), student_vals["accuracy"], label='Student Accuracy', color='red', marker='x')

    plt.title("Accuracy Comparasion")
    plt.legend()

    plt.show()
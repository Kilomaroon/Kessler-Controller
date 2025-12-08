import EasyGA as ga
import random
import numpy as np
import math
import time
import sqlite3
import os
from multiprocessing import Process
from kesslergame import Scenario, KesslerGame, GraphicsType, TrainerEnvironment
from scott_dick_controller import ScottDickController
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
                'graphics_type': GraphicsType.NoGraphics,
                'realtime_multiplier': 0,
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
        # print(e)
        return -300

    # print('Scenario eval time: '+str(time.perf_counter()-pre))
    # print(score.stop_reason)
    # print('Asteroids hit: ' + str([team.asteroids_hit for team in score.teams]))
    # print('Deaths: ' + str([team.deaths for team in score.teams]))
    # print('Mean eval time: ' + str([team.mean_eval_time for team in score.teams]))
    # score.teams[0].accuracy
    print("------------------------------------------------")
    print("asteroids hit: "+str(score.teams[0].asteroids_hit))
    print("time alive:    " + str(score.sim_time))
    print("accuracy:      "+str(score.teams[0].accuracy))
    print("fitness:       "+str((score.teams[0].asteroids_hit+score.sim_time)*score.teams[0].accuracy))

    return (score.teams[0].asteroids_hit+score.sim_time)*score.teams[0].accuracy

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
    
    batch = []
    batchsize = 3
    base_dir = os.path.dirname(os.path.abspath(__file__)) 

    for i in range(batchsize):
        db_path = os.path.join(base_dir,'batches',f'd{i}.db') 
        alg = ga.GA()
        alg.database_name = db_path
        alg.population_size = 150
        alg.generation_goal = 200
        alg.target_fitness_type = 'max'
        alg.fitness_function_impl = fitness
        alg.chromosome_impl = chromosome_function
        batch.append(Process(target=alg.evolve, args=(batchsize,)))

    for b in batch:
        b.start()

    for b in batch:
        b.join()    
        
    # get final db    
    # 

    batchdir = os.path.join(base_dir, 'batches')

    main_db = os.path.join(batchdir, 'd0.db')
    connection = sqlite3.connect(main_db)
    cursor = connection.cursor()

    cursor.execute("PRAGMA journal_mode=OFF;")
    cursor.execute("PRAGMA synchronous=OFF;")
    cursor.execute("PRAGMA foreign_keys=OFF;")

    i = 0

    for filename in os.listdir(batchdir):

        if not filename.endswith(".db"):
            continue
        if filename == "d0.db":
            continue

        file_path = os.path.join(batchdir, filename)

        alias = f"db_to_merge_{i}"
        i += 1

        cursor.execute(f"ATTACH DATABASE '{file_path}' AS {alias}")
        print("Attached", filename)

        # finalize any previous statement
        cursor.execute("SELECT 1")

        tables = cursor.execute(
            f"SELECT name FROM {alias}.sqlite_master WHERE type='table';"
        ).fetchall()

        # finalize statement
        cursor.execute("SELECT 1")

        for (table_name,) in tables:
            cursor.execute(f"""
                INSERT OR REPLACE INTO main.{table_name}
                SELECT * FROM {alias}.{table_name}
            """)

        connection.commit()

        # finalize everything before detach
        cursor.execute("SELECT 1")

        cursor.execute(f"DETACH DATABASE {alias}")
        print("Detached", filename)

    connection.commit()
    connection.close()
    print("Done")














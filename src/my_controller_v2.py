from kesslergame import KesslerController
from typing import Dict, Tuple
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import random
import numpy as np
import pygad


class my_controller(KesslerController):
    def __init__(self, chromosome = None):
        super().__init__()

        if chromosome is not None:
            self.chromosome = chromosome
            self.set_genes_from_chromosome()
        else:
            self.chromosome = self.generate_random_chromosome()
            self.set_genes_from_chromosome()

        self.generation = 0
        self.build_fuzzy_tree()

    def generate_random_chromosome(self):
        return [
            random.uniform(0.5, 2.0),  # distance_weight
            random.uniform(0.5, 2.0),  # size_weight  
            random.uniform(0.5, 2.0),  # velocity_weight
            random.uniform(0.3, 0.8),  # shoot_threshold
            random.uniform(0.4, 0.9),  # mine_threshold
            random.uniform(0.2, 0.7),  # evade_threshold
            random.uniform(100, 300),  # safe_distance
            random.uniform(0.3, 1.0)   # approach_speed
        ]

    def set_genes_from_chromosome(self):
        self.genes = {
            'distance_weight': self.chromosome[0],
            'size_weight': self.chromosome[1],
            'velocity_weight': self.chromosome[2],
            'shoot_threshold': self.chromosome[3],
            'mine_threshold': self.chromosome[4],
            'evade_threshold': self.chromosome[5],
            'safe_distance': self.chromosome[6],
            'approach_speed': self.chromosome[7]
        }

    def build_fuzzy_tree(self):
        self.build_threat_assessment_system()
        self.build_action_selection_system()
        self.build_movement_control_system()

    def build_threat_assessment_system(self):
        # inputs
        self.asteroid_distance = ctrl.Antecedant(np.arrange(0, 1000, 10), 'asteroid_distance')
        self.asteroid_relative_speed = ctrl.Antecedant(np.arrange(0, 500, 10), 'relative_speed')
        self.asteroid_size = ctrl.Antecedant(np.arrange(0, 5, 1), 'asteroid_size')

        # outputs
        self.threat_level = ctrl.Consequent(np.arrange(0, 1, 0.1), 'threat_level')

        # mumbership functions
        self.asteroid_distance['very_close'] = fuzz.trapmf(self.asteroid_distance.universe, [0, 0, 100, 200])
        self.asteroid_distance['close'] = fuzz.trimf(self.asteroid_distance.universe, [100, 250, 400])
        self.asteroid_distance['medium'] = fuzz.trimf(self.asteroid_distance.universe, [300, 500, 700])
        self.asteroid_distance['far'] = fuzz.trapmf(self.asteroid_distance.universe, [600, 800, 1000, 1000])

        self.asteroid_relative_speed['slow'] = fuzz.trapmf(self.asteroid_relative_speed.universe, [0, 0, 100, 200])
        self.asteroid_relative_speed['medium'] = fuzz.trimf(self.asteroid_relative_speed.universe, [150, 250, 350])
        self.asteroid_relative_speed['fast'] = fuzz.trapmf(self.asteroid_relative_speed.universe, [300, 400, 500, 500])

        self.asteroid_size['small'] = fuzz.trimf(self.asteroid_size.universe, [1, 1, 2])
        self.asteroid_size['medium'] = fuzz.trimf(self.asteroid_size.universe, [1.5, 2.5, 3.5])
        self.asteroid_size['large'] = fuzz.trimf(self.asteroid_size.universe, [3, 4, 4])

        self.threat_level['low'] = fuzz.trimf(self.threat_level.universe, [0, 0, 0.4])
        self.threat_level['medium'] = fuzz.trimf(self.threat_level.universe, [0.2, 0.5, 0.8])
        self.threat_level['high'] = fuzz.trimf(self.threat_level.universe, [0.6, 1, 1])

        # rules
        self.threat_rules = [
            ctrl.Rule(self.asteroid_distance['very_close'] & 
                     (self.asteroid_relative_speed['medium'] | self.asteroid_relative_speed['fast']), 
                     self.threat_level['high']),
            
            ctrl.Rule(self.asteroid_distance['close'] & self.asteroid_size['large'] & 
                     self.asteroid_relative_speed['fast'], 
                     self.threat_level['high']),
                     
            ctrl.Rule(self.asteroid_distance['medium'] & self.asteroid_size['medium'], 
                     self.threat_level['medium']),
                     
            ctrl.Rule(self.asteroid_distance['far'] | 
                     (self.asteroid_relative_speed['slow'] & self.asteroid_size['small']), 
                     self.threat_level['low'])
        ]
        
        self.threat_system = ctrl.ControlSystem(self.threat_rules)
        self.threat_sim = ctrl.ControlSystemSimulation(self.threat_system)

    def build_action_selection_system(self):
        # inputs
        self.threat_input = ctrl.Antecedent(np.arange(0, 1, 0.1), 'threat_input')
        self.aim_accuracy = ctrl.Antecedent(np.arange(0, 1, 0.1), 'aim_accuracy')
        self.available_space = ctrl.Antecedent(np.arange(0, 1, 0.1), 'available_space')

        # outputs
        self.action_priority = ctrl.Consequent(np.arange(0, 1, 0.1), 'action_priority')

        # membership functions
        self.threat_input['low'] = fuzz.trimf(self.threat_input.universe, [0, 0, 0.4])
        self.threat_input['medium'] = fuzz.trimf(self.threat_input.universe, [0.2, 0.5, 0.8])
        self.threat_input['high'] = fuzz.trimf(self.threat_input.universe, [0.6, 1, 1])
        
        self.aim_accuracy['poor'] = fuzz.trimf(self.aim_accuracy.universe, [0, 0, 0.5])
        self.aim_accuracy['good'] = fuzz.trimf(self.aim_accuracy.universe, [0.3, 0.7, 1])
        self.aim_accuracy['excellent'] = fuzz.trimf(self.aim_accuracy.universe, [0.8, 1, 1])
        
        self.available_space['tight'] = fuzz.trimf(self.available_space.universe, [0, 0, 0.4])
        self.available_space['adequate'] = fuzz.trimf(self.available_space.universe, [0.2, 0.5, 0.8])
        self.available_space['open'] = fuzz.trimf(self.available_space.universe, [0.6, 1, 1])
        
        self.action_priority['evade'] = fuzz.trimf(self.action_priority.universe, [0, 0, 0.4])
        self.action_priority['mine'] = fuzz.trimf(self.action_priority.universe, [0.2, 0.5, 0.8])
        self.action_priority['shoot'] = fuzz.trimf(self.action_priority.universe, [0.6, 1, 1])

        # rules
        self.action_rules = [
            # High threat -> evade
            ctrl.Rule(self.threat_input['high'] & self.available_space['tight'], 
                     self.action_priority['evade']),
            
            # Good aim + medium threat -> shoot  
            ctrl.Rule(self.threat_input['medium'] & self.aim_accuracy['good'], 
                     self.action_priority['shoot']),
                     
            # Low threat + tight space -> mine for defense
            ctrl.Rule(self.threat_input['low'] & self.available_space['tight'], 
                     self.action_priority['mine']),
                     
            # Default to shooting if well-aimed
            ctrl.Rule(self.aim_accuracy['excellent'], 
                     self.action_priority['shoot'])
        ]
        
        self.action_system = ctrl.ControlSystem(self.action_rules)
        self.action_sim = ctrl.ControlSystemSimulation(self.action_system)

    def build_movement_control_system(self):
        # inputs
        self.selected_action = ctrl.Antecedent(np.arange(0, 1, 0.1), 'selected_action')
        self.asteroid_direction = ctrl.Antecedent(np.arange(-math.pi, math.pi, 0.1), 'asteroid_direction')
        
        # outputs
        self.thrust_output = ctrl.Consequent(np.arange(-1, 1, 0.1), 'thrust_output')
        self.turn_output = ctrl.Consequent(np.arange(-1, 1, 0.1), 'turn_output')
        
        # membership functions
        self.selected_action['evade'] = fuzz.trimf(self.selected_action.universe, [0, 0, 0.4])
        self.selected_action['mine'] = fuzz.trimf(self.selected_action.universe, [0.2, 0.5, 0.8]) 
        self.selected_action['shoot'] = fuzz.trimf(self.selected_action.universe, [0.6, 1, 1])
        
        self.asteroid_direction['left'] = fuzz.trapmf(self.asteroid_direction.universe, 
                                                     [-math.pi, -math.pi, -math.pi/4, 0])
        self.asteroid_direction['front'] = fuzz.trimf(self.asteroid_direction.universe, 
                                                     [-math.pi/4, 0, math.pi/4])
        self.asteroid_direction['right'] = fuzz.trapmf(self.asteroid_direction.universe, 
                                                      [0, math.pi/4, math.pi, math.pi])
        
        self.thrust_output['reverse'] = fuzz.trimf(self.thrust_output.universe, [-1, -1, 0])
        self.thrust_output['coast'] = fuzz.trimf(self.thrust_output.universe, [-0.5, 0, 0.5])
        self.thrust_output['forward'] = fuzz.trimf(self.thrust_output.universe, [0, 1, 1])
        
        self.turn_output['left'] = fuzz.trimf(self.turn_output.universe, [-1, -1, 0])
        self.turn_output['straight'] = fuzz.trimf(self.turn_output.universe, [-0.3, 0, 0.3])
        self.turn_output['right'] = fuzz.trimf(self.turn_output.universe, [0, 1, 1])
        
        # rules
        self.movement_rules = [
            # Evade: move away from threat
            ctrl.Rule(self.selected_action['evade'] & self.asteroid_direction['left'],
                     (self.thrust_output['forward'], self.turn_output['right'])),
            ctrl.Rule(self.selected_action['evade'] & self.asteroid_direction['right'], 
                     (self.thrust_output['forward'], self.turn_output['left'])),
            ctrl.Rule(self.selected_action['evade'] & self.asteroid_direction['front'],
                     (self.thrust_output['reverse'], self.turn_output['left'])),
                     
            # Shoot: approach while aiming
            ctrl.Rule(self.selected_action['shoot'] & self.asteroid_direction['front'],
                     (self.thrust_output['forward'], self.turn_output['straight'])),
                     
            # Mine: position defensively  
            ctrl.Rule(self.selected_action['mine'],
                     (self.thrust_output['coast'], self.turn_output['straight']))
        ]
        
        self.movement_system = ctrl.ControlSystem(self.movement_rules)
        self.movement_sim = ctrl.ControlSystemSimulation(self.movement_system)

    def evaluate_asteroid_threat(self, asteroid, ship_state):
        ship_pos =ship_state.position
        asteroid_pos = asteroid.position # check
        asteroid_vel = asteroid.velocity
        
        distance = math.dist(ship_pos, asteroid_pos)

        ship_vel = ship_state.velocity
        relative_vel = math.dist(asteroid_vel, ship_vel)

        size_factor = asteroid.size / 4.0

        weighted_threat = (
            (1.0 / max(distance, 1)) * self.genes['distance_weight'] +
            relative_vel * self.genes['velocity_weight'] + 
            size_factor * self.genes['size_weight']
        )

        return weighted_threat, distance, relative_vel, size_factor
    
    def calculate_aim_accuracy(self, ship_state, asteroid):
        ship_pos = ship_state.position 
        ship_heading = math.radians(ship_state.heading)

        asteroid_pos = asteroid.position

        dx = asteroid_pos[0] - ship_pos[0]
        dy = asteroid_pos[1] - ship_pos[1]
        angle_to_asteroid = math.atan2(dy, dx)

        angle_diff = abs(angle_to_asteroid - ship_heading)
        angle_diff = min(angle_diff, 2 * math.pi - angle_diff)

        accuracy = 1.0 - (angle_diff / math.pi)
        return max(0, accuracy)
    
    def calculate_available_space(self, ship_state, game_state):
        ship_pos = ship_state.position
        map_size = game_state.map_size

        # distance to the nearest edge
        dist_to_left = ship_pos[0]
        dist_to_right = map_size[0] - ship_pos[0]
        dist_to_top = ship_pos[1]
        dist_to_bottom = map_size[1] - ship_pos[1]

        min_dist_to_edge = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)

        # resclare to [0, 1]
        max_possible_dist = max(map_size[0], map_size[1]) / 2
        space_ratio = min_dist_to_edge / max_possible_dist
        
        return space_ratio
    
    def actions(self, ship_state: ShipState, game_state: GameState): 
        asteroids = game_state.asteroids 
        ship_pos = ship_state.position

        if not asteroids:
            return (0.0, 0.0, False, False)
        
        # find the most threatening asteroid
        highest_threat = -1
        threat_asteroid = None
        threat_data = None

        for asteroid in asteroids:
            threat, distance, rel_speed, size = self.evaluate_asteroid_threat(asteroid, ship_state)
            if threat > highest_threat:
                highest_threat = threat
                threat_asteroid = asteroid
                threat_data = (distance, rel_speed, size)
        
        if not threat_asteroid:
            return (0.0, 0.0, False, False)
        
        # threat accessment
        distance, rel_speed, size = threat_data
        self.threat_sim.input['asteroid_distance'] = distance
        self.threat_sim.input['asteroid_relative_speed'] = rel_speed
        self.threat_sim.input['asteroid_size'] = size

        try:
            self.threat_sim.compute()
            threat_level = self.threat_sim.output['threat_level']
        except:
            threat_level = 0.5

        # action selection
        aim_accuracy = self.calculate_aim_accuracy(ship_state, threat_asteroid)
        available_space = self.calculate_available_space(ship_state, game_state)
        
        self.action_sim.input['threat_input'] = threat_level
        self.action_sim.input['aim_accuracy'] = aim_accuracy
        self.action_sim.input['available_space'] = available_space
        
        try:
            self.action_sim.compute()
            action_priority = self.action_sim.output['action_priority']
        except:
            action_priority = 0.5

        # movement control
        asteroid_pos = threat_asteroid.position 
        dx = asteroid_pos[0] - ship_pos[0]
        dy = asteroid_pos[1] - ship_pos[1]
        asteroid_direction = math.atan2(dy, dx)
        
        self.movement_sim.input['selected_action'] = action_priority
        self.movement_sim.input['asteroid_direction'] = asteroid_direction
        
        try:
            self.movement_sim.compute()
            thrust = self.movement_sim.output['thrust_output']
            turn = self.movement_sim.output['turn_output']
        except:
            thrust, turn = 0.0, 0.0

        # convert fuzzy outputs to game commands
        thrust_range = ship_state.thrust_range
        max_thrust = max(thrust_range)
        actual_thrust = thrust * max_thrust
        
        turn_range = ship_state.turn_rate_range
        max_turn = max(turn_range)
        actual_turn = turn * max_turn

        # fire action
        fire = (action_priority > self.genes['shoot_threshold'] and 
                aim_accuracy > 0.7 and 
                ship_state.can_fire) 
        
        mine = (action_priority > self.genes['mine_threshold'] and 
                threat_level > 0.6 and 
                available_space < 0.3 and
                ship_state.can_deploy_mine)
        
        # apply genetic parameters
        actual_thrust = actual_thrust * self.genes['approach_speed']

        # check output range
        actual_thrust = max(thrust_range[0], min(thrust_range[1], actual_thrust))
        actual_turn = max(turn_range[0], min(turn_range[1], actual_turn))

        return (actual_thrust, actual_turn, fire, mine)
        
    def mutate_genes(self, mutation_rate = 0.1):
        for key in self.genes:
            if random.random() < mutation_rate:
                if 'weight' in key:
                    self.genes[key] *= random.uniform(0.8, 1.2)
                elif 'threshold' in key:
                    self.genes[key] = max(0.1, min(0.9, 
                        self.genes[key] + random.uniform(-0.1, 0.1)))
                else:
                    self.genes[key] *= random.uniform(0.9, 1.1)
        
        self.generation += 1

    def get_chromosome(self):
        return self.chromosome
    
    @property
    def name(self) -> str:
        return f"My Controller v{self.generation}"
    
def create_genetic_algorithm():

    def fitness_func(ga_instance, solution, solution_idx):
        controller = my_controller(chromosome=solution)
        fitness_score = run_game_simulation(controller) # fix
        return fitness_score

    gene_space = [
        {'low': 0.5, 'high': 2.0},    # distance_weight
        {'low': 0.5, 'high': 2.0},    # size_weight
        {'low': 0.5, 'high': 2.0},    # velocity_weight
        {'low': 0.3, 'high': 0.8},    # shoot_threshold
        {'low': 0.4, 'high': 0.9},    # mine_threshold  
        {'low': 0.2, 'high': 0.7},    # evade_threshold
        {'low': 100, 'high': 300},    # safe_distance
        {'low': 0.3, 'high': 1.0}     # approach_speed
    ]

    ga_instance = pygad.GA(
        num_generations=50,
        num_parents_mating=4,
        fitness_func=fitness_func,
        sol_per_pop=10,
        num_genes=8,
        gene_space=gene_space,
        parent_selection_type="sss",
        keep_parents=1,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=10
    )

    return ga_instance

def get_evolved_controller():
    ga_instance = create_genetic_algorithm()
    ga_instance.run()

    best_solution, best_fitness, _ = ga_instance.best_solution()
    print(f"Best fitness: {best_fitness}")

    return my_controller(chromosome=best_solution)






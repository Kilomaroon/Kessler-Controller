from kesslergame import KesslerController 
from typing import Dict, Tuple
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np
import pygad

MAX_SPEED = 60
MAX_DISTANCE = 1000
MAX_TURN = 5
MAX_ASTEROIDS = 500
MAX_LIVES = 3
MAX_MINES = 3
BASIC_RANGE = np.arange(0,1,0.001)

class MyController(KesslerController):
    def __init__(self):
            
        self.set_parameters_from_chromosome()
        
    
    def get_default_chromosome(self):
        """Return default chromosome values for genetic evolution"""
        return [

        ]
    
    def set_parameters_from_chromosome(self):
        return 0

    def actions(self, ship_state: Dict, game_state) -> Tuple[float, float, bool, bool]:

        return (float(1), float(1), bool(True), bool(True))

    @property
    def name(self) -> str:
        return f"Group <INSERT GROUP NUMBER HERE> controller"
    
    def make_rules(self):
        # -------------- LAYERS -----------------
        # Layer 0
        mine_risk = FIS('mine_risk') # risk for mine to hit us
        asteroid_agression = FIS('asteroid_agression') # how badly we want to target asteroids
        opponent_agression = FIS('opponent_agression') # how badly we want to target other player
        bullet_confidence = FIS('bullet_confidence') # expectation for bullet to hit target

        # Layer 1
        mine_confidence = FIS('mine_confidence') # expectation for mine to hit target
        priority = FIS('priority') # if our priority should be evasion or attack
        threat = FIS('threat') # how scared we are of death
        turn = FIS('turn') # how necessary we think turning is and in what direction

        # Output layer
        thrust = FIS('thrust')
        turn_rate = FIS('turn_rate')
        fire = FIS('fire')
        drop_mine = FIS('drop_mine')

        # -------------- LAYER 0 INPUTS ----------------
        mine_risk.add_antecedent(np.arange(0,MAX_LIVES,1),'lives_remaining') # how many lives do we have left
        mine_risk.add_antecedent(np.arange(0,MAX_ASTEROIDS,1),'nearby_asteroids') # asteroids near us within some radius r
        mine_risk.add_antecedent(np.arange(0,MAX_SPEED,0.01),'speed') # how fast we are currently going
        mine_risk.add_antecedent(np.arange(0,0,0.1,'mine_fired')) # have we fired a mine lately

        asteroid_agression.add_antecedent(np.arange(0,MAX_ASTEROIDS,1),'nearby_asteroids')
        asteroid_agression.add_antecedent(np.arange(0,MAX_DISTANCE,0.01),'nearest_asteroid') # how far away the nearest asteroid is
        asteroid_agression.add_antecedent(np.arange(0,1,0.1),'mine_fired') 

        opponent_agression.add_antecedent(np.arange(0,MAX_DISTANCE,0.01),'player_distance') # how far away the other player is
        opponent_agression.add_antecedent(np.arange(0,MAX_LIVES,1),'opponent_lives') # not sure if you can pull this val. nix it if not ig
        opponent_agression.add_antecedent(np.arange(0,MAX_MINES,1),'mines_remaining') # how many mines we have left

        bullet_confidence.add_antecedent(np.arange(0,1.0,0.001),'bullet_error') # some value you contrive to refer to error of the previous shots, feel free to change the range
        bullet_confidence.add_antecedent(np.arange(0,MAX_DISTANCE,0.01),'nearest_asteroid') # distance to the nearest asteroid
        bullet_confidence.add_antecedent(np.arange(0,100,0.1),'accurancy') # current accuracy score

        mine_risk.consequent = (BASIC_RANGE,mine_risk.name)
        asteroid_agression.consequent = (BASIC_RANGE,asteroid_agression.name)
        opponent_agression.consequent = (BASIC_RANGE,opponent_agression.name)
        bullet_confidence.consequent = (BASIC_RANGE,bullet_confidence.name)

        # -------------- LAYER 1 INPUTS -----------------
        mine_confidence.add_antecedent()
        

        
    
class FIS:
    Nodes = []

    def __init__(self,name):
        self.antecedents = []    # list of all antecedents to current node
        self.rules = []          # list of all rules for current node
        self.name = name         # node name - make sure consistent
        FIS.Nodes.append(self)   # add self to list of all nodes
        self.value = 1             # the defuzzified output from the system

    def add_antecedent(self, range, name):
        self.antecedents.append(ctrl.Antecendents(range,name))

    def set_concequent(self, c):
        self.consequent = ctrl.Consequent(c[0], c[1])

    def build_tree():
        for node in FIS.Nodes:
            if node.antecedents:
                node.control = ctrl.ControlSystem(node.rules)
                node.sim = ctrl.ControlSystemSimulation(node.control)
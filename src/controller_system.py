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
MAX_THRUST = 10

class MyController(KesslerController):
    def __init__(self):
        self.rules_values = []
        self.make_tree()    
        self.set_parameters_from_chromosome()
        
    
    def get_default_chromosome(self):
        """Return default chromosome values for genetic evolution"""
        return [

        ]
    
    def set_parameters_from_chromosome(self):
        self.rules_values = [1,2,3,4,5,6,7,8,9,10]

    def actions(self, ship_state: Dict, game_state) -> Tuple[float, float, bool, bool]:

        return (float(1), float(1), bool(True), bool(True))

    @property
    def name(self) -> str:
        return f"Group <INSERT GROUP NUMBER HERE> controller"
    
    def make_tree(self):
        # -------------- LAYERS -----------------
        # Layer 0
        self.risk = FIS('mine_risk') # how badly we want to escape
        self.agression = FIS('opponent_agression') # how badly we want to attack
        self.mine_confidence = FIS('mine_confidence') # can we get away from a mine if we place one
        
        # Layer 1
        self.atk_turn = FIS('atk_turn') # turning needed to atk
        self.priority = FIS('priority') # if our priority should be evasion or attack
        self.run_turn = FIS('run_turn') # turning needed for escape

        # Output layer
        self.thrust = FIS('thrust')
        self.fire = FIS('fire')
        self.drop_mine = FIS('drop_mine')

        # -------------- LAYER 0 INPUTS ----------------
        self.risk.add_antecedent(np.arange(0,MAX_ASTEROIDS,1),'nearby_asteroids') # asteroids near us within some radius r
        self.risk.add_antecedent(np.arange(0,MAX_DISTANCE,0.01),'nearest_enemy') # how far away the nearest asteroid or mine is

        self.agression.add_antecedent(np.arange(-1*math.pi/30,math.pi/30,0.1),'theta_delta_atk') # angle between desired target and current direction
        self.agression.add_antecedent(np.arange(0,MAX_MINES,1),'mines_remaining') # how many mines we have left
        
        self.mine_confidence.add_antecedent(np.arange(0,MAX_SPEED,0.01),'current_speed') # time it takes for a mine to reach desired
        self.mine_confidence.add_antecedent(np.arange(0,MAX_MINES,1),'mines_remaining') #number of mines left
        self.mine_confidence.add_antecedent(np.arange(0,50,1),'enemies_ahead') # how many enemies in front of us (within a certain range)

        self.risk.consequent = (BASIC_RANGE,self.risk.name)
        self.agression.consequent = (BASIC_RANGE,self.agression.name)
        self.mine_confidence.consequent = (BASIC_RANGE,self.priority.name)
       

        # -------------- LAYER 1 INPUTS -----------------
        self.priority.add_antecedent(BASIC_RANGE,self.agression.name)
        self.priority.add_antecedent(BASIC_RANGE,self.risk.name)

        self.run_turn.add_antecedent(BASIC_RANGE,self.priority.name)
        self.run_turn.add_antecedent(np.arange(-1*math.pi/30,math.pi/30,0.1), 'theta_delta_def') # angle between us and nearest enemy 

        self.atk_turn.add_antecedent(BASIC_RANGE,self.priority.name)
        self.atk_turn.add_antecedent(np.arange(0,1.0,0.002),'bullet_time') # time it takes for a bullet to reach desired
        self.atk_turn.add_antecedent(np.arange(-1*math.pi/30,math.pi/30,0.1), 'theta_delta_atk') # angle between desired target and current direction

        self.atk_turn.consequent = (BASIC_RANGE,self.atk_turn.name)
        self.priority.consequent = (BASIC_RANGE,self.priority.name)
        self.run_turn.consequent = (BASIC_RANGE,self.run_turn.name)

        # ------------ OUTPUT LAYER -----------------------

        self.thrust.add_antecedent(BASIC_RANGE, self.risk.name)
        self.thrust.add_antecedent(np.arange(0,50,1),'enemies_ahead')
        
        self.fire.add_antecedent(BASIC_RANGE, self.priority.name)
        self.fire.add_antecedent(BASIC_RANGE, self.atk_turn.name)

        self.drop_mine.add_antecedent(BASIC_RANGE, self.mine_confidence.name)
        self.drop_mine.add_antecedent(BASIC_RANGE, self.priority.name)

        self.thrust.consequent = (np.arange(0,MAX_THRUST,0.01),self.thrust.name)
        self.fire.consequent = (BASIC_RANGE,self.fire.name)
        self.drop_mine.add_antecedent = (BASIC_RANGE, self.drop_mine.name)

    def makerules(self):
        # -------------- LAYER 0 ----------------
        # nearby_asteroids
        (self.risk.antecedents[0])['none'] = fuzz.trimf(self.risk.antecedents[0].universe, [0,self.rules_values[0],self.rules_values[1]])
        (self.risk.antecedents[0])['few'] = fuzz.trimf(self.risk.antecedents[0].universe, [self.rules_values[2],self.rules_values[3],self.rules_values[4]])
        (self.risk.antecedents[0])['many'] = fuzz.trimf(self.risk.antecedents[0].universe, [self.rules_values[5],self.rules_values[6],MAX_ASTEROIDS])

        # nearest_enemy
        (self.risk.antecedents[1])['close'] = fuzz.trimf(self.risk.antecedents[1].universe, [0,self.rules_values[7],self.rules_values[8]])
        (self.risk.antecedents[1])['far'] = fuzz.trimf(self.risk.antecedents[1].universe, [self.rules_values[9],self.rules_values[10],MAX_DISTANCE])

        # theta_delta_atk
        (self.agression.antecedents[0])['NL'] = fuzz.trimf(self.agression.antecedents[0], [self.rules_values[11],self.rules_values[12],self.rules_values[13]])
        (self.agression.antecedents[0])['NM'] = fuzz.trimf(self.agression.antecedents[0], [self.rules_values[14],self.rules_values[15],self.rules_values[16]])
        (self.agression.antecedents[0])['NS'] = fuzz.trimf(self.agression.antecedents[0], [self.rules_values[17],self.rules_values[18],self.rules_values[19]])
        (self.agression.antecedents[0])['PL'] = fuzz.trimf(self.agression.antecedents[0], [self.rules_values[20],self.rules_values[21],self.rules_values[22]])
        (self.agression.antecedents[0])['PM'] = fuzz.trimf(self.agression.antecedents[0], [self.rules_values[23],self.rules_values[24],self.rules_values[25]])
        (self.agression.antecedents[0])['PS'] = fuzz.trimf(self.agression.antecedents[0], [self.rules_values[26],self.rules_values[27],self.rules_values[28]])

        




        

        
    
class FIS:
    Nodes = []

    def __init__(self,name):
        self.antecedents = []    # list of all antecedents to current node
        self.rules = []          # list of all rules for current node
        self.name = name         # node name - make sure consistent
        FIS.Nodes.append(self)   # add self to list of all nodes
        self.value = 1             # the defuzzified output from the system

    def add_antecedent(self, range, name):
        self.antecedents.append(ctrl.Antecedent(range,name))

    def set_concequent(self, c):
        self.consequent = ctrl.Consequent(c[0], c[1])

    def build_tree():
        for node in FIS.Nodes:
            if node.antecedents:
                node.control = ctrl.ControlSystem(node.rules)
                node.sim = ctrl.ControlSystemSimulation(node.control)
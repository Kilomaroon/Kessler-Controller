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
TOP = 1
BASIC_RANGE = np.arange(0,TOP,0.001)
MAX_THRUST = 10
MAX_ENEMIES = 50

class MyController(KesslerController):
    def __init__(self):
        self.rules_values = []
        self.set_parameters_from_chromosome()
        self.make_tree()
        self.make_ranges()  
        
        
    
    def get_default_chromosome(self):
        """Return default chromosome values for genetic evolution"""
        return [

        ]
    
    def set_parameters_from_chromosome(self):
        self.rules_values = np.arange(0,29,1)

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
        self.mine_confidence.add_antecedent(np.arange(0,MAX_ENEMIES,1),'enemies_ahead') # how many enemies in front of us (within a certain range)

        self.risk.set_concequent(BASIC_RANGE,self.risk.name)
        self.agression.set_concequent(BASIC_RANGE,self.agression.name)
        self.mine_confidence.set_concequent(BASIC_RANGE,self.priority.name)
       

        # -------------- LAYER 1 INPUTS -----------------
        self.priority.add_antecedent(BASIC_RANGE,self.agression.name)
        self.priority.add_antecedent(BASIC_RANGE,self.risk.name)

        self.run_turn.add_antecedent(BASIC_RANGE,self.priority.name)
        self.run_turn.add_antecedent(np.arange(-1*math.pi/30,math.pi/30,0.1), 'theta_delta_def') # angle between us and nearest enemy 

        self.atk_turn.add_antecedent(BASIC_RANGE,self.priority.name)
        self.atk_turn.add_antecedent(BASIC_RANGE,'bullet_time') # time it takes for a bullet to reach desired
        self.atk_turn.add_antecedent(np.arange(-1*math.pi/30,math.pi/30,0.1), 'theta_delta_atk') # angle between desired target and current direction

        self.atk_turn.set_concequent(BASIC_RANGE,self.atk_turn.name)
        self.priority.set_concequent(BASIC_RANGE,self.priority.name)
        self.run_turn.set_concequent(BASIC_RANGE,self.run_turn.name)

        # ------------ OUTPUT LAYER -----------------------

        self.thrust.add_antecedent(BASIC_RANGE, self.risk.name)
        self.thrust.add_antecedent(np.arange(0,MAX_ENEMIES,1),'enemies_ahead')
        
        self.fire.add_antecedent(BASIC_RANGE, self.priority.name)
        self.fire.add_antecedent(BASIC_RANGE, self.atk_turn.name)

        self.drop_mine.add_antecedent(BASIC_RANGE, self.mine_confidence.name)
        self.drop_mine.add_antecedent(BASIC_RANGE, self.priority.name)

        self.thrust.set_concequent(np.arange(0,MAX_THRUST,0.01),self.thrust.name)
        self.fire.set_concequent(BASIC_RANGE,self.fire.name)
        self.drop_mine.set_concequent(BASIC_RANGE, self.drop_mine.name)

    def make_ranges(self):
        # -------------- LAYER 0 ----------------
        # nearby_asteroids
        (self.risk.antecedents[0])['none'] = fuzz.trimf(self.risk.antecedents[0].universe, [0,self.rules_values[0],self.rules_values[1]])
        (self.risk.antecedents[0])['few'] = fuzz.trimf(self.risk.antecedents[0].universe, [self.rules_values[2],self.rules_values[3],self.rules_values[4]])
        (self.risk.antecedents[0])['many'] = fuzz.trimf(self.risk.antecedents[0].universe, [self.rules_values[5],self.rules_values[6],MAX_ASTEROIDS])

        # nearest_enemy
        (self.risk.antecedents[1])['close'] = fuzz.trimf(self.risk.antecedents[1].universe, [0,self.rules_values[7],self.rules_values[8]])
        (self.risk.antecedents[1])['far'] = fuzz.trimf(self.risk.antecedents[1].universe, [self.rules_values[9],self.rules_values[10],MAX_DISTANCE])

        ## RISK
        self.risk.consequent['safe'] = fuzz.trimf(self.risk.consequent.universe, [0,self.rules_values[51],self.rules_values[52]])
        self.risk.consequent['danger'] = fuzz.trimf(self.risk.consequent.universe, [self.rules_values[53],self.rules_values[54],TOP])

        # theta_delta_atk
        (self.agression.antecedents[0])['NL'] = fuzz.trimf(self.agression.antecedents[0].universe, [-1*math.pi/30,self.rules_values[12],self.rules_values[13]])
        (self.agression.antecedents[0])['NM'] = fuzz.trimf(self.agression.antecedents[0].universe, [self.rules_values[14],self.rules_values[15],self.rules_values[16]])
        (self.agression.antecedents[0])['NS'] = fuzz.trimf(self.agression.antecedents[0].universe, [self.rules_values[17],self.rules_values[18],self.rules_values[19]])
        (self.agression.antecedents[0])['PL'] = fuzz.trimf(self.agression.antecedents[0].universe, [self.rules_values[20],self.rules_values[21],self.rules_values[22]])
        (self.agression.antecedents[0])['PM'] = fuzz.trimf(self.agression.antecedents[0].universe, [self.rules_values[23],self.rules_values[24],self.rules_values[25]])
        (self.agression.antecedents[0])['PS'] = fuzz.trimf(self.agression.antecedents[0].universe, [self.rules_values[26],self.rules_values[27],math.pi/30])
       
       # mines_remaining
        (self.agression.antecedents[1])['few']  = fuzz.trimf(self.agression.antecedents[1].universe, [0,self.rules_values[30],self.rules_values[31]])
        (self.agression.antecedents[1])['many'] = fuzz.trimf(self.agression.antecedents[1].universe, [self.rules_values[32],self.rules_values[33],MAX_MINES])

        ## AGRESSION

        self.agression.consequent['shy'] = fuzz.trimf(self.agression.consequent.universe, [0,self.rules_values[45],self.rules_values[46]])
        self.agression.consequent['calm'] = fuzz.trimf(self.agression.consequent.universe, [self.rules_values[46],self.rules_values[47],self.rules_values[48]])
        self.agression.consequent['angry'] = fuzz.trimf(self.agression.consequent.universe, [self.rules_values[49],self.rules_values[50],TOP])   

       # current_speed
        (self.mine_confidence.antecedents[0])['slow'] = fuzz.trimf(self.mine_confidence.antecedents[0].universe, [0,self.rules_values[34],self.rules_values[35]])
        (self.mine_confidence.antecedents[0])['normal'] = fuzz.trimf(self.mine_confidence.antecedents[0].universe, [self.rules_values[36],self.rules_values[37],self.rules_values[38]])
        (self.mine_confidence.antecedents[0])['fast'] = fuzz.trimf(self.mine_confidence.antecedents[0].universe, [self.rules_values[39],self.rules_values[40],MAX_SPEED])

       # mines_remaining 
        (self.mine_confidence.antecedents[1])['few']  = (self.agression.antecedents[1])['few']
        (self.mine_confidence.antecedents[1])['many'] = (self.agression.antecedents[1])['many']

        # enemies_ahead
        (self.mine_confidence.antecedents[2])['few']  = fuzz.trimf(self.mine_confidence.antecedents[2].universe, [0,self.rules_values[41],self.rules_values[42]])
        (self.mine_confidence.antecedents[2])['many'] = fuzz.trimf(self.mine_confidence.antecedents[2].universe, [self.rules_values[43],self.rules_values[44],MAX_ENEMIES])

        ## MINE CONF
        self.mine_confidence.consequent['risky'] = fuzz.trimf(self.mine_confidence.consequent.universe, [0,self.rules_values[30],self.rules_values[31]])
        self.mine_confidence.consequent['confident'] = fuzz.trimf(self.mine_confidence.consequent.universe, [self.rules_values[49],self.rules_values[50],TOP])  
       
        # -------------- LAYER 1  -----------------
        
        # agression
        (self.priority.antecedents[0])['shy'] = fuzz.trimf(self.priority.antecedents[0].universe, [0,self.rules_values[45],self.rules_values[46]])
        (self.priority.antecedents[0])['calm'] = fuzz.trimf(self.priority.antecedents[0].universe, [self.rules_values[46],self.rules_values[47],self.rules_values[48]])
        (self.priority.antecedents[0])['angry'] = fuzz.trimf(self.priority.antecedents[0].universe, [self.rules_values[49],self.rules_values[50],TOP])

        # risk
        (self.priority.antecedents[1])['safe'] = fuzz.trimf(self.priority.antecedents[1].universe, [0,self.rules_values[51],self.rules_values[52]])
        (self.priority.antecedents[1])['danger'] = fuzz.trimf(self.priority.antecedents[1].universe, [self.rules_values[53],self.rules_values[54],TOP])

        ## PRIORITY
        self.priority.consequent['attack'] = fuzz.trimf(self.priority.consequent.universe, [0,self.rules_values[30],self.rules_values[31]])
        self.priority.consequent['flee'] = fuzz.trimf(self.priority.consequent.universe, [self.rules_values[49],self.rules_values[50],TOP]) 

        # priority
        (self.run_turn.antecedents[0])['attack'] = fuzz.trimf(self.run_turn.antecedents[0].universe, [0,self.rules_values[30],self.rules_values[31]])
        (self.run_turn.antecedents[0])['flee'] = fuzz.trimf(self.run_turn.antecedents[0].universe, [self.rules_values[49],self.rules_values[50],TOP]) 

        # theta_delta_def
        (self.run_turn.antecedents[1])['NL'] = fuzz.trimf(self.run_turn.antecedents[1].universe, [-1*math.pi/30,self.rules_values[12],self.rules_values[13]])
        (self.run_turn.antecedents[1])['NM'] = fuzz.trimf(self.run_turn.antecedents[1].universe, [self.rules_values[14],self.rules_values[15],self.rules_values[16]])
        (self.run_turn.antecedents[1])['NS'] = fuzz.trimf(self.run_turn.antecedents[1].universe, [self.rules_values[17],self.rules_values[18],self.rules_values[19]])
        (self.run_turn.antecedents[1])['PL'] = fuzz.trimf(self.run_turn.antecedents[1].universe, [self.rules_values[20],self.rules_values[21],self.rules_values[22]])
        (self.run_turn.antecedents[1])['PM'] = fuzz.trimf(self.run_turn.antecedents[1].universe, [self.rules_values[23],self.rules_values[24],self.rules_values[25]])
        (self.run_turn.antecedents[1])['PS'] = fuzz.trimf(self.run_turn.antecedents[1].universe, [self.rules_values[26],self.rules_values[27],math.pi/30])
        
        ## RUN TURN
        self.run_turn.consequent['NL'] = fuzz.trimf(self.run_turn.consequent.universe, [-1*math.pi/30,self.rules_values[12],self.rules_values[13]])
        self.run_turn.consequent['NM'] = fuzz.trimf(self.run_turn.consequent.universe, [self.rules_values[14],self.rules_values[15],self.rules_values[16]])
        self.run_turn.consequent['NS'] = fuzz.trimf(self.run_turn.consequent.universe, [self.rules_values[17],self.rules_values[18],self.rules_values[19]])
        self.run_turn.consequent['PL'] = fuzz.trimf(self.run_turn.consequent.universe, [self.rules_values[17],self.rules_values[18],self.rules_values[19]])
        self.run_turn.consequent['PM'] = fuzz.trimf(self.run_turn.consequent.universe, [self.rules_values[17],self.rules_values[18],self.rules_values[19]])
        self.run_turn.consequent['PS'] = fuzz.trimf(self.run_turn.consequent.universe, [self.rules_values[26],self.rules_values[27],math.pi/30])

        # priority
        (self.atk_turn.antecedents[0])['attack'] = (self.run_turn.antecedents[0])['attack'] 
        (self.atk_turn.antecedents[0])['flee'] = (self.run_turn.antecedents[0])['flee'] 

        # delta atk
        (self.atk_turn.antecedents[0])['NL'] = fuzz.trimf(self.atk_turn.antecedents[0].universe, [-1*math.pi/30,self.rules_values[12],self.rules_values[13]])
        (self.atk_turn.antecedents[0])['NM'] = fuzz.trimf(self.atk_turn.antecedents[0].universe, [self.rules_values[14],self.rules_values[15],self.rules_values[16]])
        (self.atk_turn.antecedents[0])['NS'] = fuzz.trimf(self.atk_turn.antecedents[0].universe, [self.rules_values[17],self.rules_values[18],self.rules_values[19]])
        (self.atk_turn.antecedents[0])['PL'] = fuzz.trimf(self.atk_turn.antecedents[0].universe, [self.rules_values[20],self.rules_values[21],self.rules_values[22]])
        (self.atk_turn.antecedents[0])['PM'] = fuzz.trimf(self.atk_turn.antecedents[0].universe, [self.rules_values[23],self.rules_values[24],self.rules_values[25]])
        (self.atk_turn.antecedents[0])['PS'] = fuzz.trimf(self.atk_turn.antecedents[0].universe, [self.rules_values[26],self.rules_values[27],math.pi/30])

        # bullet time
        (self.atk_turn.antecedents[1])['slow'] = fuzz.trimf(self.atk_turn.antecedents[1].universe, [0,self.rules_values[12],self.rules_values[13]])
        (self.atk_turn.antecedents[1])['med'] = fuzz.trimf(self.atk_turn.antecedents[1].universe, [self.rules_values[14],self.rules_values[15],self.rules_values[16]])
        (self.atk_turn.antecedents[1])['fast'] = fuzz.trimf(self.atk_turn.antecedents[1].universe, [self.rules_values[17],self.rules_values[18],TOP])

        ## ATK_TURN
        self.atk_turn.consequent['NL'] = fuzz.trimf(self.atk_turn.consequent.universe, [-1*math.pi/30,self.rules_values[12],self.rules_values[13]])
        self.atk_turn.consequent['NM'] = fuzz.trimf(self.atk_turn.consequent.universe, [self.rules_values[14],self.rules_values[15],self.rules_values[16]])
        self.atk_turn.consequent['NS'] = fuzz.trimf(self.atk_turn.consequent.universe, [self.rules_values[17],self.rules_values[18],self.rules_values[19]])
        self.atk_turn.consequent['PL'] = fuzz.trimf(self.atk_turn.consequent.universe, [self.rules_values[17],self.rules_values[18],self.rules_values[19]])
        self.atk_turn.consequent['PM'] = fuzz.trimf(self.atk_turn.consequent.universe, [self.rules_values[17],self.rules_values[18],self.rules_values[19]])
        self.atk_turn.consequent['PS'] = fuzz.trimf(self.atk_turn.consequent.universe, [self.rules_values[26],self.rules_values[27],math.pi/30])

        # ------------ OUTPUT LAYER -----------------------

        # risk
        (self.thrust.antecedents[0])['safe'] = fuzz.trimf(self.thrust.antecedents[0].universe, [0,self.rules_values[51],self.rules_values[52]])
        (self.thrust.antecedents[0])['danger'] = fuzz.trimf(self.thrust.antecedents[0].universe, [self.rules_values[53],self.rules_values[54],TOP])

        # enemies ahead
        (self.thrust.antecedents[1])['few'] = (self.mine_confidence.antecedents[2])['few'] 
        (self.thrust.antecedents[1])['many'] = (self.mine_confidence.antecedents[1])['many']

        #@ THRUST
        self.thrust.consequent['go'] = fuzz.trimf(self.thrust.consequent.universe, [0,self.rules_values[51],self.rules_values[52]])
        self.thrust.consequent['stop'] = fuzz.trimf(self.thrust.consequent.universe, [0,self.rules_values[51],TOP])

        # priority
        (self.fire.antecedents[0])['attack'] = (self.run_turn.antecedents[0])['attack'] 
        (self.fire.antecedents[0])['flee'] = (self.run_turn.antecedents[0])['flee'] 

        # atk_turn
        (self.fire.antecedents[1])['NL'] = fuzz.trimf(self.fire.antecedents[1].universe, [-1*math.pi/30,self.rules_values[12],self.rules_values[13]])
        (self.fire.antecedents[1])['NM'] = fuzz.trimf(self.fire.antecedents[1].universe, [self.rules_values[14],self.rules_values[15],self.rules_values[16]])
        (self.fire.antecedents[1])['NS'] = fuzz.trimf(self.fire.antecedents[1].universe, [self.rules_values[17],self.rules_values[18],self.rules_values[19]])
        (self.fire.antecedents[1])['PL'] = fuzz.trimf(self.fire.antecedents[1].universe, [self.rules_values[17],self.rules_values[18],self.rules_values[19]])
        (self.fire.antecedents[1])['PM'] = fuzz.trimf(self.fire.antecedents[1].universe, [self.rules_values[17],self.rules_values[18],self.rules_values[19]])
        (self.fire.antecedents[1])['PS'] = fuzz.trimf(self.fire.antecedents[1].universe, [self.rules_values[26],self.rules_values[27],math.pi/30])

        # FIRE
        self.fire.consequent['shoot'] =  fuzz.trimf(self.fire.consequent.universe, [0,self.rules_values[51],self.rules_values[52]])
        self.fire.consequent['hold']  = fuzz.trimf(self.fire.consequent.universe, [self.rules_values[53],self.rules_values[54],TOP])

        # mine confidence
        (self.drop_mine.antecedents[0])['risky'] = fuzz.trimf(self.mine_confidence.consequent.universe, [0,self.rules_values[30],self.rules_values[31]])
        (self.drop_mine.antecedents[0])['confident'] = fuzz.trimf(self.mine_confidence.consequent.universe, [0,self.rules_values[30],TOP])

        # priority
        (self.drop_mine.antecedents[1])['attack'] = (self.run_turn.antecedents[0])['attack'] 
        (self.drop_mine.antecedents[1])['flee'] = (self.run_turn.antecedents[0])['flee'] 

        # DROP MINE
        self.drop_mine.consequent['shoot'] =  fuzz.trimf(self.drop_mine.consequent.universe, [0,self.rules_values[51],self.rules_values[52]])
        self.drop_mine.consequent['hold']  = fuzz.trimf(self.drop_mine.consequent.universe, [self.rules_values[53],self.rules_values[54],TOP])       

    def add_rules(self):
        # TODO: ADD RULES

        for node in FIS.Nodes:
            if node.antecedents:
                node.control = ctrl.ControlSystem(node.rules)
                node.sim = ctrl.ControlSystemSimulation(node.control)
                    





        

        
    
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

    def set_concequent(self,range,name):
        self.consequent = ctrl.Consequent(range, name)

    def build_tree():
        for node in FIS.Nodes:
            if node.antecedents:
                node.control = ctrl.ControlSystem(node.rules)
                node.sim = ctrl.ControlSystemSimulation(node.control)
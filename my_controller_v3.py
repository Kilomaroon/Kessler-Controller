from kesslergame import KesslerController 
from typing import Dict, Tuple
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np

class MyController(KesslerController):
    def __init__(self):
        super().__init__()
        self.eval_frames = 0             # counter to track how many frames the controller has processed
        self.last_mine_time = -100       # to track last mine drop time

        self.build_targeting_system()    # aiming and shooting
        self.build_movement_system()     # thrust and positioning
        self.build_survival_system()     # emergency evasion
        
    def build_targeting_system(self):
        """
        inputs: bullet travel time, aim error angle, asteroid size
        outputs: turn rate, fire command
        """
        # inputs 
        bullet_time = ctrl.Antecedent(np.arange(0, 3.0, 0.01), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-math.pi, math.pi, 0.05), 'theta_delta')
        asteroid_size = ctrl.Antecedent(np.arange(1, 5, 1), 'asteroid_size')
        
        # Outputs 
        ship_turn = ctrl.Consequent(np.arange(-180, 180, 1), 'ship_turn')
        ship_fire = ctrl.Consequent(np.arange(0, 1, 0.1), 'ship_fire')
    
        
        # how long until bullet hits (0-3s)
        bullet_time['instant'] = fuzz.trimf(bullet_time.universe, [0, 0, 0.05])
        bullet_time['very_fast'] = fuzz.trimf(bullet_time.universe, [0, 0.05, 0.1])
        bullet_time['fast'] = fuzz.trimf(bullet_time.universe, [0.05, 0.1, 0.3])
        bullet_time['medium'] = fuzz.trimf(bullet_time.universe, [0.1, 0.3, 0.8])
        bullet_time['slow'] = fuzz.trimf(bullet_time.universe, [0.3, 0.8, 1.5])
        bullet_time['very_slow'] = fuzz.smf(bullet_time.universe, 0.8, 3.0)

        # angle difference from perfect aim (-pi-pi)
        theta_delta['sharp_left'] = fuzz.zmf(theta_delta.universe, -math.pi, -math.pi/3)
        theta_delta['left'] = fuzz.trimf(theta_delta.universe, [-math.pi/2, -math.pi/6, -math.pi/12])
        theta_delta['slight_left'] = fuzz.trimf(theta_delta.universe, [-math.pi/6, -math.pi/12, 0])
        theta_delta['perfect'] = fuzz.trimf(theta_delta.universe, [-math.pi/24, 0, math.pi/24])
        theta_delta['slight_right'] = fuzz.trimf(theta_delta.universe, [0, math.pi/12, math.pi/6])
        theta_delta['right'] = fuzz.trimf(theta_delta.universe, [math.pi/12, math.pi/6, math.pi/2])
        theta_delta['sharp_right'] = fuzz.smf(theta_delta.universe, math.pi/3, math.pi)

        # asteroid size
        asteroid_size['small'] = fuzz.trimf(asteroid_size.universe, [1, 1, 2])
        asteroid_size['medium'] = fuzz.trimf(asteroid_size.universe, [1.5, 2.5, 3.5])
        asteroid_size['large'] = fuzz.trimf(asteroid_size.universe, [3, 4, 4])
        
        # turn rate (-180-180)
        ship_turn['sharp_left'] = fuzz.trimf(ship_turn.universe, [-180, -180, -120])
        ship_turn['left'] = fuzz.trimf(ship_turn.universe, [-150, -90, -30])
        ship_turn['slight_left'] = fuzz.trimf(ship_turn.universe, [-60, -30, 0])
        ship_turn['fine_tune'] = fuzz.trimf(ship_turn.universe, [-15, 0, 15])
        ship_turn['slight_right'] = fuzz.trimf(ship_turn.universe, [0, 30, 60])
        ship_turn['right'] = fuzz.trimf(ship_turn.universe, [30, 90, 150])
        ship_turn['sharp_right'] = fuzz.trimf(ship_turn.universe, [120, 180, 180])
            
        # fire command (True/False)
        ship_fire['no'] = fuzz.trimf(ship_fire.universe, [0, 0, 0.6])
        ship_fire['yes'] = fuzz.trimf(ship_fire.universe, [0.4, 1, 1])

        self.targeting_control = ctrl.ControlSystem()
        
        # bullet_time = instant
        self.targeting_control.addrule(ctrl.Rule(bullet_time['instant'] & theta_delta['sharp_left'], (ship_turn['sharp_left'], ship_fire['yes'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['instant'] & theta_delta['left'], (ship_turn['left'], ship_fire['yes'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['instant'] & theta_delta['slight_left'], (ship_turn['slight_left'], ship_fire['yes'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['instant'] & theta_delta['perfect'], (ship_turn['fine_tune'], ship_fire['yes'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['instant'] & theta_delta['slight_right'], (ship_turn['slight_right'], ship_fire['yes'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['instant'] & theta_delta['right'], (ship_turn['right'], ship_fire['yes'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['instant'] & theta_delta['sharp_right'], (ship_turn['sharp_right'], ship_fire['yes'])))
        # bullet_time = very_fast
        self.targeting_control.addrule(ctrl.Rule(bullet_time['very_fast'] & theta_delta['sharp_left'], (ship_turn['sharp_left'], ship_fire['no'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['very_fast'] & theta_delta['left'], (ship_turn['left'], ship_fire['yes'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['very_fast'] & theta_delta['slight_left'], (ship_turn['slight_left'], ship_fire['yes'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['very_fast'] & theta_delta['perfect'], (ship_turn['fine_tune'], ship_fire['yes'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['very_fast'] & theta_delta['slight_right'], (ship_turn['slight_right'], ship_fire['yes'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['very_fast'] & theta_delta['right'], (ship_turn['right'], ship_fire['yes'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['very_fast'] & theta_delta['sharp_right'], (ship_turn['sharp_right'], ship_fire['no'])))
        # bullet_time = fast
        self.targeting_control.addrule(ctrl.Rule(bullet_time['fast'] & theta_delta['sharp_left'], (ship_turn['sharp_left'], ship_fire['no'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['fast'] & theta_delta['left'], (ship_turn['left'], ship_fire['no'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['fast'] & theta_delta['slight_left'], (ship_turn['slight_left'], ship_fire['yes'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['fast'] & theta_delta['perfect'], (ship_turn['fine_tune'], ship_fire['yes'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['fast'] & theta_delta['slight_right'], (ship_turn['slight_right'], ship_fire['yes'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['fast'] & theta_delta['right'], (ship_turn['right'], ship_fire['no'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['fast'] & theta_delta['sharp_right'], (ship_turn['sharp_right'], ship_fire['no'])))
        # bullet_time = medium
        self.targeting_control.addrule(ctrl.Rule(bullet_time['medium'] & theta_delta['sharp_left'], (ship_turn['sharp_left'], ship_fire['no'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['medium'] & theta_delta['left'], (ship_turn['left'], ship_fire['no'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['medium'] & theta_delta['slight_left'], (ship_turn['slight_left'], ship_fire['no'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['medium'] & theta_delta['perfect'], (ship_turn['fine_tune'], ship_fire['yes'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['medium'] & theta_delta['slight_right'], (ship_turn['slight_right'], ship_fire['no'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['medium'] & theta_delta['right'], (ship_turn['right'], ship_fire['no'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['medium'] & theta_delta['sharp_right'], (ship_turn['sharp_right'], ship_fire['no'])))
        # bullet_time = slow
        self.targeting_control.addrule(ctrl.Rule(bullet_time['slow'] & theta_delta['sharp_left'], (ship_turn['sharp_left'], ship_fire['no'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['slow'] & theta_delta['left'], (ship_turn['left'], ship_fire['no'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['slow'] & theta_delta['slight_left'], (ship_turn['slight_left'], ship_fire['no'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['slow'] & theta_delta['perfect'], (ship_turn['fine_tune'], ship_fire['no'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['slow'] & theta_delta['slight_right'], (ship_turn['slight_right'], ship_fire['no'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['slow'] & theta_delta['right'], (ship_turn['right'], ship_fire['no'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['slow'] & theta_delta['sharp_right'], (ship_turn['sharp_right'], ship_fire['no'])))
        # bullet_time = very_slow
        self.targeting_control.addrule(ctrl.Rule(bullet_time['very_slow'] & theta_delta['sharp_left'], (ship_turn['sharp_left'], ship_fire['no'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['very_slow'] & theta_delta['left'], (ship_turn['left'], ship_fire['no'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['very_slow'] & theta_delta['slight_left'], (ship_turn['slight_left'], ship_fire['no'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['very_slow'] & theta_delta['perfect'], (ship_turn['fine_tune'], ship_fire['no'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['very_slow'] & theta_delta['slight_right'], (ship_turn['slight_right'], ship_fire['no'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['very_slow'] & theta_delta['right'], (ship_turn['right'], ship_fire['no'])))
        self.targeting_control.addrule(ctrl.Rule(bullet_time['very_slow'] & theta_delta['sharp_right'], (ship_turn['sharp_right'], ship_fire['no'])))
        
        # large asteroids
        self.targeting_control.addrule(ctrl.Rule(asteroid_size['large'] & theta_delta['slight_left'], (ship_turn['slight_left'], ship_fire['yes'])))
        self.targeting_control.addrule(ctrl.Rule(asteroid_size['large'] & theta_delta['slight_right'], (ship_turn['slight_right'], ship_fire['yes'])))
        
        self.targeting_sim = ctrl.ControlSystemSimulation(self.targeting_control)

    def build_movement_system(self):
        """
        inputs: closest threat distance, threat approach speed, edge distance
        outputs: thrust command
        """
        # input
        threat_distance = ctrl.Antecedent(np.arange(0, 1000, 10), 'threat_distance')
        approach_speed = ctrl.Antecedent(np.arange(-500, 500, 10), 'approach_speed')
        edge_distance = ctrl.Antecedent(np.arange(0, 500, 10), 'edge_distance')
        
        # output
        ship_thrust = ctrl.Consequent(np.arange(-500, 500, 10), 'ship_thrust')
        
        # threat distance 
        threat_distance['safe'] = fuzz.trapmf(threat_distance.universe, [0, 0, 150, 300])
        threat_distance['caution'] = fuzz.trimf(threat_distance.universe, [200, 350, 500])
        threat_distance['danger'] = fuzz.smf(threat_distance.universe, 400, 1000)
        
        # threat approach speed (-ve = moving away, +ve = approaching)
        approach_speed['moving_away'] = fuzz.trapmf(approach_speed.universe, [-500, -500, -100, 0])
        approach_speed['stable'] = fuzz.trimf(approach_speed.universe, [-50, 0, 50])
        approach_speed['approaching_fast'] = fuzz.smf(approach_speed.universe, 50, 500)
        
        # distance from map boundary
        edge_distance['at_edge'] = fuzz.trapmf(edge_distance.universe, [0, 0, 30, 80])
        edge_distance['near_edge'] = fuzz.trimf(edge_distance.universe, [50, 100, 180])
        edge_distance['safe_center'] = fuzz.smf(edge_distance.universe, 150, 500)
        
        # thrust (-ve = moving away, +ve = approaching))
        ship_thrust['full_reverse'] = fuzz.trimf(ship_thrust.universe, [-500, -500, -300])
        ship_thrust['reverse'] = fuzz.trimf(ship_thrust.universe, [-400, -200, 0])
        ship_thrust['stop'] = fuzz.trimf(ship_thrust.universe, [-100, 0, 100])
        ship_thrust['forward'] = fuzz.trimf(ship_thrust.universe, [0, 200, 400])
        ship_thrust['full_forward'] = fuzz.trimf(ship_thrust.universe, [300, 500, 500])
                        
        self.movement_control = ctrl.ControlSystem()
    
        # threat_distance = dange
        self.movement_control.addrule(ctrl.Rule(threat_distance['danger'] & approach_speed['approaching_fast'], ship_thrust['full_reverse']))
        self.movement_control.addrule(ctrl.Rule(threat_distance['danger'] & approach_speed['stable'], ship_thrust['reverse']))
        self.movement_control.addrule(ctrl.Rule(threat_distance['danger'] & approach_speed['moving_away'], ship_thrust['stop']))
        #  threat_distance = caution
        self.movement_control.addrule(ctrl.Rule(threat_distance['caution'] & approach_speed['approaching_fast'], ship_thrust['stop']))
        self.movement_control.addrule(ctrl.Rule(threat_distance['caution'] & approach_speed['stable'], ship_thrust['forward']))
        self.movement_control.addrule(ctrl.Rule(threat_distance['caution'] & approach_speed['moving_away'], ship_thrust['forward']))
        # threat_distance = safe
        self.movement_control.addrule(ctrl.Rule(threat_distance['safe'] & approach_speed['approaching_fast'], ship_thrust['forward']))
        self.movement_control.addrule(ctrl.Rule(threat_distance['safe'] & approach_speed['stable'], ship_thrust['forward']))
        self.movement_control.addrule(ctrl.Rule(threat_distance['safe'] & approach_speed['moving_away'], ship_thrust['forward']))
        
        # edge avoidance (high priority)
        self.movement_control.addrule(ctrl.Rule(edge_distance['at_edge'], ship_thrust['full_forward']))
        self.movement_control.addrule(ctrl.Rule(edge_distance['near_edge'], ship_thrust['forward']))
        self.movement_control.addrule(ctrl.Rule(edge_distance['safe_center'], ship_thrust['stop']))
        
        self.movement_sim = ctrl.ControlSystemSimulation(self.movement_control)

    def build_survival_system(self):
        """
        inputs: threat level, health status  
        outputs: survival override
        """
        # input
        threat_level = ctrl.Antecedent(np.arange(0, 1, 0.1), 'threat_level')
        health_status = ctrl.Antecedent(np.arange(0, 1, 0.1), 'health_status')
        
        # output
        survival_override = ctrl.Consequent(np.arange(0, 1, 0.1), 'survival_override')
        
        # threat level (0-1)
        threat_level['low'] = fuzz.trimf(threat_level.universe, [0, 0, 0.4])
        threat_level['medium'] = fuzz.trimf(threat_level.universe, [0.3, 0.5, 0.7])
        threat_level['high'] = fuzz.trimf(threat_level.universe, [0.6, 1, 1])
        
        # health status (0-1)
        health_status['critical'] = fuzz.trimf(health_status.universe, [0, 0, 0.3])
        health_status['healthy'] = fuzz.trimf(health_status.universe, [0.2, 1, 1])
        
        # survival override (0-1)
        survival_override['normal'] = fuzz.trimf(survival_override.universe, [0, 0, 0.3])
        survival_override['evade'] = fuzz.trimf(survival_override.universe, [0.2, 0.5, 0.8])
        survival_override['panic'] = fuzz.trimf(survival_override.universe, [0.7, 1, 1])
        
        self.survival_control = ctrl.ControlSystem()
        
        # high threat | critical health
        self.survival_control.addrule(ctrl.Rule(threat_level['high'] | health_status['critical'], survival_override['panic']))
        # medium threat & critical health
        self.survival_control.addrule(ctrl.Rule(threat_level['medium'] & health_status['critical'], survival_override['panic']))
        # medium threat & healthy
        self.survival_control.addrule(ctrl.Rule(threat_level['medium'] & health_status['healthy'], survival_override['evade']))
        # low threat & healthy
        self.survival_control.addrule(ctrl.Rule(threat_level['low'] & health_status['healthy'], survival_override['normal']))
        
        self.survival_sim = ctrl.ControlSystemSimulation(self.survival_control)

    def get_movement_thrust(self, ship_state, game_state):
        """get thrust from movement fuzzy system"""
        try:
            # calculate threat metrics
            threats = self.assess_threats(game_state["asteroids"], ship_state)
            closest_threat = min(threats, key=lambda x: x['distance']) if threats else None
            
            # calculate distance to map boundary
            map_size = game_state["map_size"]
            ship_pos = ship_state["position"]
            edge_distances = [ship_pos[0], map_size[0]-ship_pos[0], ship_pos[1], map_size[1]-ship_pos[1]]
            min_edge_distance = min(edge_distances)
            
            if not closest_threat or closest_threat['distance'] > 400:
                return 0.0

            if closest_threat:
                self.movement_sim.input['threat_distance'] = closest_threat['distance']
                self.movement_sim.input['approach_speed'] = closest_threat['approach_speed']
            else:
                self.movement_sim.input['threat_distance'] = 1000  # safe distance
                self.movement_sim.input['approach_speed'] = -500   # moving away
            
            self.movement_sim.input['edge_distance'] = min_edge_distance
            self.movement_sim.compute()
            return -float(self.movement_sim.output['ship_thrust'])
        except Exception:
            return 0.0

    def apply_survival_override(self, ship_state, game_state, thrust, fire, aim_error):
        """apply survival system overrides"""
        try:
            # threat score (0-1)
            threats = self.assess_threats(game_state["asteroids"], ship_state)
            max_threat = max([t['threat_score'] for t in threats]) if threats else 0
            
            # health score (0-1)
            health = ship_state["lives_remaining"] / 3.0
            
            # input: threat level, health status
            # output: survival override (0-1)
            self.survival_sim.input['threat_level'] = max_threat
            self.survival_sim.input['health_status'] = health
            self.survival_sim.compute()
            survival_mode = self.survival_sim.output['survival_override']
            
            # extreme danger
            if survival_mode > 0.8: 
                thrust = -400.0
                fire = fire and (aim_error < 0.1) # only shoot if very accurate
            # moderate danger
            elif survival_mode > 0.5: 
                thrust = -300.0
                fire = fire and (aim_error < 0.15) # reduce shooting
                
            return thrust, fire
        except Exception:
            return thrust, fire

    def should_drop_mine(self, ship_state, game_state):
        """determine if we should drop a mine"""
        try:
            threats = self.assess_threats(game_state["asteroids"], ship_state)
            close_threats = [t for t in threats if t['distance'] < 200 and t['approach_speed'] > 100]
            
            # calculate edge distance
            map_size = game_state["map_size"]
            ship_pos = ship_state["position"]
            edge_distances = [ship_pos[0], map_size[0]-ship_pos[0], ship_pos[1], map_size[1]-ship_pos[1]]
            min_edge_distance = min(edge_distances)
            
            ship_vel = ship_state["velocity"]
            ship_speed = math.sqrt(ship_vel[0]**2 + ship_vel[1]**2)

            mine_conditions = (
                len(close_threats) >= 2 and               # multiple threats
                min_edge_distance > 150 and               # safe from edges  
                ship_speed > 50 and                       # not stationary
                not self.recent_mine_dropped()            # not too frequent
            )

            if mine_conditions:
                self.last_mine_time = self.eval_frames
                return True
        
            return False
        except Exception:
            return False

    def recent_mine_dropped(self):
        """Check if mine was dropped recently to prevent spamming"""
        if not hasattr(self, 'last_mine_time'):
            self.last_mine_time = -100  
        return (self.eval_frames - self.last_mine_time) < 60  # about 2s

    def assess_threats(self, asteroids, ship_state):
        """threat assessment for movement and survival systems"""
        ship_pos = ship_state["position"]
        ship_vel = ship_state["velocity"]
        threats = []
        
        for asteroid in asteroids:
            distance = math.dist(ship_pos, asteroid['position']) 
            
            # threat score based on distance
            threat_score = 1.0 / (distance + 0.01)
            threat_score = min(1.0, threat_score)
            
            # calculate approach speed 
            to_asteroid = [asteroid['position'][0]-ship_pos[0], asteroid['position'][1]-ship_pos[1]] # vector from ship to asteroid
            distance_norm = math.sqrt(to_asteroid[0]**2 + to_asteroid[1]**2) # magnitude
            if distance_norm > 0:
                relative_vel = [asteroid['velocity'][0]-ship_vel[0], asteroid['velocity'][1]-ship_vel[1]]
                # the dot product giving the projection of relative velocity onto the ship-asteroid direction
                approach_speed = (relative_vel[0]*to_asteroid[0] + relative_vel[1]*to_asteroid[1]) / distance_norm
            else:
                approach_speed = 0
            
            threats.append({
                'asteroid': asteroid,
                'distance': distance,
                'approach_speed': approach_speed,
                'threat_score': threat_score
            })
        
        return threats

    def calculate_intercept(self, ship_pos, asteroid):
        """
        calculate intercept point and time using law of cosines approach
        return intercept position and time to intercept
        """
        asteroid_pos = asteroid['position']
        asteroid_vel = asteroid['velocity']
        bullet_speed = 800
        
        # asteroid-ship vector and angle
        asteroid_ship_x = ship_pos[0] - asteroid_pos[0]
        asteroid_ship_y = ship_pos[1] - asteroid_pos[1]
        asteroid_ship_theta = math.atan2(asteroid_ship_y, asteroid_ship_x)
        
        # asteroid direction and angle difference
        asteroid_direction = math.atan2(asteroid_vel[1], asteroid_vel[0])
        theta2 = asteroid_ship_theta - asteroid_direction
        cos_theta2 = math.cos(theta2)
        
        # asteroid speed and distance
        asteroid_speed = math.sqrt(asteroid_vel[0]**2 + asteroid_vel[1]**2)
        distance = math.sqrt(asteroid_ship_x**2 + asteroid_ship_y**2)
        
        # solve quadratic equation for intercept time
        a = asteroid_speed**2 - bullet_speed**2
        b = 2 * distance * asteroid_speed * cos_theta2
        c = distance**2
        
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return None, float('inf')
        
        t1 = (-b + math.sqrt(discriminant)) / (2*a)
        t2 = (-b - math.sqrt(discriminant)) / (2*a)
        
        # choose smallest positive intercept time
        if t1 > t2:
            bullet_t = t2 if t2 >= 0 else t1
        else:
            bullet_t = t1 if t1 >= 0 else t2
        
        if bullet_t <= 0:
            return None, float('inf')
        
        # intercept position 
        intrcpt_x = asteroid_pos[0] + asteroid_vel[0] * (bullet_t + 1/30)
        intrcpt_y = asteroid_pos[1] + asteroid_vel[1] * (bullet_t + 1/30)
        
        return (intrcpt_x, intrcpt_y), bullet_t

    def find_best_target(self, asteroids, ship_state):
        """
        find closest asteroid as best target 
        returns best asteroid, intercept position, intercept time
        """
        ship_pos = ship_state["position"]
        best_target = None
        min_distance = float('inf')
        
        for asteroid in asteroids:
            distance = math.dist(ship_pos, asteroid['position'])
            if distance < min_distance:
                min_distance = distance
                best_target = asteroid
        
        if best_target is None:
            return None, None, float('inf')
        
        intercept_pos, intercept_time = self.calculate_intercept(ship_pos, best_target)
        return best_target, intercept_pos, intercept_time

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        asteroids = game_state["asteroids"]
        if not asteroids:
            return (0.0, 0.0, False, False)
        
        ship_pos = ship_state["position"]
        ship_heading_rad = math.radians(ship_state["heading"])
        
        # find best target (closest asteroid)
        best_asteroid, intercept_pos, intercept_time = self.find_best_target(asteroids, ship_state)
        
        if not best_asteroid or intercept_time == float('inf'):
            thrust = self.get_movement_thrust(ship_state, game_state)
            #return (0.0, 0.0, False, False)
        
        # calculate theta_delta
        dx = intercept_pos[0] - ship_pos[0]
        dy = intercept_pos[1] - ship_pos[1]
        target_angle = math.atan2(dy, dx)
        theta_delta = target_angle - ship_heading_rad
        theta_delta = (theta_delta + math.pi) % (2 * math.pi) - math.pi 
        
        try:
            self.targeting_sim.input['bullet_time'] = min(intercept_time, 3.0)
            self.targeting_sim.input['theta_delta'] = theta_delta
            self.targeting_sim.input['asteroid_size'] = best_asteroid['size']
            self.targeting_sim.compute()
            turn_rate = float(self.targeting_sim.output['ship_turn'])
            fire_output = self.targeting_sim.output['ship_fire']
            fire = bool(fire_output > 0.5)
        except Exception:
            # proportional control
            turn_rate = float(theta_delta * 180 / math.pi)
            fire = bool(abs(theta_delta) < 0.2 and intercept_time < 2.0)
        
        # movement
        thrust = self.get_movement_thrust(ship_state, game_state)
        # survival 
        thrust, fire = self.apply_survival_override(ship_state, game_state, thrust, fire, theta_delta)
        # mine
        drop_mine = self.should_drop_mine(ship_state, game_state)

        # thrust = 0.0
        # drop_mine = False
        
        return (float(thrust), float(turn_rate), bool(fire), bool(drop_mine))

    @property
    def name(self) -> str:
        return "My Controller"
from kesslergame import KesslerController 
from typing import Dict, Tuple
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np

class MyController(KesslerController):
    def __init__(self,chrom_vals):
        super().__init__()
        self.eval_frames = 0             # counter to track how many frames the controller has processed
        self.last_mine_time = 0       # to track last mine drop time

        self.build_targeting_system(chrom_vals)    # aiming and shooting
        self.build_movement_system(chrom_vals)     # thrust and positioning
        self.build_survival_system(chrom_vals)     # emergency evasion
        
    def build_targeting_system(self, chrom_vals):
        """
        inputs: bullet travel time, aim error angle, asteroid size
        outputs: turn rate, fire command
        """
        # inputs 
        bullet_time = ctrl.Antecedent(np.arange(0, 3.0, 0.01), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-math.pi, math.pi, 0.01), 'theta_delta')
        asteroid_size = ctrl.Antecedent(np.arange(1, 5, 1), 'asteroid_size')
        
        # Outputs 
        ship_turn = ctrl.Consequent(np.arange(-180, 180, 1), 'ship_turn')
        ship_fire = ctrl.Consequent(np.arange(0, 1, 0.1), 'ship_fire')
    
        
        # how long until bullet hits (0-3s)
        bullet_time['instant'] = fuzz.trimf(bullet_time.universe,           [0, 0, chrom_vals[0]])
        bullet_time['very_fast'] = fuzz.trimf(bullet_time.universe,         [0, chrom_vals[0], chrom_vals[1]])
        bullet_time['fast'] = fuzz.trimf(bullet_time.universe,              [chrom_vals[0], chrom_vals[1], chrom_vals[2]])
        bullet_time['medium'] = fuzz.trimf(bullet_time.universe,            [chrom_vals[2], chrom_vals[3], chrom_vals[4]])
        bullet_time['slow'] = fuzz.trimf(bullet_time.universe,              [chrom_vals[3], chrom_vals[4], chrom_vals[5]])
        bullet_time['very_slow'] = fuzz.trimf(bullet_time.universe,         [chrom_vals[4], chrom_vals[5], 3.0])

        # angle difference from perfect aim (-pi-pi)
        theta_delta['sharp_left'] = fuzz.trimf(theta_delta.universe, [-math.pi, chrom_vals[6],chrom_vals[7]])
        theta_delta['left'] = fuzz.trimf(theta_delta.universe, [chrom_vals[7], chrom_vals[8], chrom_vals[9]])
        theta_delta['slight_left'] = fuzz.trimf(theta_delta.universe, [chrom_vals[8], chrom_vals[9], chrom_vals[10]])
        theta_delta['perfect'] = fuzz.trimf(theta_delta.universe, [chrom_vals[10], chrom_vals[11], chrom_vals[12]])
        theta_delta['slight_right'] = fuzz.trimf(theta_delta.universe, [chrom_vals[11], chrom_vals[12], chrom_vals[13]])
        theta_delta['right'] = fuzz.trimf(theta_delta.universe, [chrom_vals[12], chrom_vals[13], chrom_vals[14]])
        theta_delta['sharp_right'] = fuzz.trimf(theta_delta.universe, [chrom_vals[14],chrom_vals[15] ,math.pi])

        # asteroid size
        asteroid_size['small'] = fuzz.trimf(asteroid_size.universe, [1, chrom_vals[16], chrom_vals[17]])
        asteroid_size['medium'] = fuzz.trimf(asteroid_size.universe, [chrom_vals[17], chrom_vals[18], chrom_vals[19]])
        asteroid_size['large'] = fuzz.trimf(asteroid_size.universe, [chrom_vals[19], chrom_vals[20], 4])
        
        # turn rate (-180-180)
        ship_turn['sharp_left'] = fuzz.trimf(ship_turn.universe, [-180, chrom_vals[21], chrom_vals[22]])
        ship_turn['left'] = fuzz.trimf(ship_turn.universe, [chrom_vals[22], chrom_vals[23], chrom_vals[24]])
        ship_turn['slight_left'] = fuzz.trimf(ship_turn.universe, [chrom_vals[24], chrom_vals[25], chrom_vals[26]])
        ship_turn['fine_tune'] = fuzz.trimf(ship_turn.universe, [chrom_vals[25], chrom_vals[26], chrom_vals[27]])
        ship_turn['slight_right'] = fuzz.trimf(ship_turn.universe, [chrom_vals[27], chrom_vals[28], chrom_vals[29]])
        ship_turn['right'] = fuzz.trimf(ship_turn.universe, [chrom_vals[29], chrom_vals[30], chrom_vals[31]])
        ship_turn['sharp_right'] = fuzz.trimf(ship_turn.universe, [chrom_vals[31], chrom_vals[32], 180])
            
        # fire command (True/False)
        ship_fire['no'] = fuzz.trimf(ship_fire.universe, [0, chrom_vals[33], chrom_vals[34]])
        ship_fire['yes'] = fuzz.trimf(ship_fire.universe, [chrom_vals[34], chrom_vals[35], 1])

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
        self.targeting_control.addrule(ctrl.Rule(bullet_time['slow'] & theta_delta['perfect'], (ship_turn['fine_tune'], ship_fire['yes'])))
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
        self.targeting_control.addrule(ctrl.Rule(asteroid_size['large'] & theta_delta['slight_left'],ship_fire['yes']))
        self.targeting_control.addrule(ctrl.Rule(asteroid_size['large'] & theta_delta['slight_right'],ship_fire['yes']))
        self.targeting_control.addrule(ctrl.Rule(asteroid_size['small'] & theta_delta['sharp_left'],ship_fire['no']))
        self.targeting_control.addrule(ctrl.Rule(asteroid_size['small'] & theta_delta['sharp_right'],ship_fire['no']))


        
        self.targeting_sim = ctrl.ControlSystemSimulation(self.targeting_control)

    def build_movement_system(self, chrom_vals):
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
        threat_distance['safe'] = fuzz.trapmf(threat_distance.universe, [0, chrom_vals[36],chrom_vals[37], chrom_vals[38]])
        threat_distance['caution'] = fuzz.trimf(threat_distance.universe, [chrom_vals[37], chrom_vals[38], chrom_vals[39]])
        threat_distance['danger'] = fuzz.smf(threat_distance.universe, chrom_vals[39], 1000)
        
        # threat approach speed (-ve = moving away, +ve = approaching)
        approach_speed['moving_away'] = fuzz.trapmf(approach_speed.universe, [-500, chrom_vals[40], chrom_vals[41],chrom_vals[42]])
        approach_speed['stable'] = fuzz.trimf(approach_speed.universe, [chrom_vals[41],chrom_vals[42], chrom_vals[43]])
        approach_speed['approaching_fast'] = fuzz.smf(approach_speed.universe, chrom_vals[42], chrom_vals[44])
        
        # distance from map boundary
        edge_distance['at_edge'] = fuzz.trapmf(edge_distance.universe, [0, chrom_vals[45], chrom_vals[46], chrom_vals[47]])
        edge_distance['near_edge'] = fuzz.trimf(edge_distance.universe, [chrom_vals[46], chrom_vals[47], chrom_vals[48]])
        edge_distance['safe_center'] = fuzz.smf(edge_distance.universe, chrom_vals[48], 500)
        
        # thrust (-ve = moving away, +ve = approaching))
        ship_thrust['full_reverse'] = fuzz.trimf(ship_thrust.universe, [-500, chrom_vals[49], chrom_vals[50]])
        ship_thrust['reverse'] = fuzz.trimf(ship_thrust.universe, [chrom_vals[50], chrom_vals[51], chrom_vals[52]])
        ship_thrust['stop'] = fuzz.trimf(ship_thrust.universe, [chrom_vals[51], chrom_vals[52], chrom_vals[53]])
        ship_thrust['forward'] = fuzz.trimf(ship_thrust.universe, [chrom_vals[52], chrom_vals[53], chrom_vals[54]])
        ship_thrust['full_forward'] = fuzz.trimf(ship_thrust.universe, [chrom_vals[53], chrom_vals[54], 500])
                        
        self.movement_control = ctrl.ControlSystem()
    
        # threat_distance = danger
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

    def build_survival_system(self, chrom_vals):
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
        threat_level['low'] = fuzz.trimf(threat_level.universe, [0, chrom_vals[55],chrom_vals[56]])
        threat_level['medium'] = fuzz.trimf(threat_level.universe, [chrom_vals[56], chrom_vals[57], chrom_vals[58]])
        threat_level['high'] = fuzz.trimf(threat_level.universe, [chrom_vals[58], chrom_vals[59], 1])
        
        # health status (0-1)
        health_status['critical'] = fuzz.trimf(health_status.universe, [0, chrom_vals[60], chrom_vals[61]])
        health_status['healthy'] = fuzz.trimf(health_status.universe, [chrom_vals[61], chrom_vals[62], 1])
        
        # survival override (0-1)
        survival_override['normal'] = fuzz.trimf(survival_override.universe, [0, chrom_vals[63], chrom_vals[64]])
        survival_override['evade'] = fuzz.trimf(survival_override.universe, [chrom_vals[64], chrom_vals[65], chrom_vals[66]])
        survival_override['panic'] = fuzz.trimf(survival_override.universe, [chrom_vals[65], chrom_vals[67], 1])
        
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
            threats = self.assess_threats(game_state["asteroids"], ship_state, game_state['mines'])
            map_size = game_state["map_size"]
            ship_pos = ship_state["position"]

            edge_distances = [
                ship_pos[0],
                map_size[0] - ship_pos[0],
                ship_pos[1],
                map_size[1] - ship_pos[1],
            ]
            min_edge_distance = min(edge_distances)

            if threats:
                closest = min(threats, key=lambda t: t['distance'])
                dist = closest['distance']
                approach_speed = closest['approach_speed'] 
            else:
                dist = 1000
                approach_speed = -500

            
            if dist < 150:    # adjust this for what distance is too close to the asteroids
        
                # print(f"[ESCAPE] dist={dist:.1f} -> hard thrust")
                return -200.0

            
            self.movement_sim.input['threat_distance'] = dist
            self.movement_sim.input['approach_speed'] = approach_speed
            self.movement_sim.input['edge_distance'] = min_edge_distance + 50

            self.movement_sim.compute()
            thrust = float(self.movement_sim.output['ship_thrust'])

            
            thrust *= 2.0

            # print(f"[MOVE] dist={dist:.1f} speed={approach_speed:.1f} thrust={thrust:.1f}")
            return thrust

        except Exception as e:
            # print("Movement error:", e)
            return 0.0

    def apply_survival_override(self, ship_state, game_state, thrust, fire, aim_error):
        """apply survival system overrides"""
        try:
            # threat score (0-1)
            threats = self.assess_threats(game_state["asteroids"], ship_state, game_state['mines'])
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
            
            threats = self.assess_threats(game_state["asteroids"], ship_state, game_state['mines'])
            close_threats = [t for t in threats if t['distance'] < 100]
            
            # calculate edge distance
            map_size = game_state["map_size"]
            ship_pos = ship_state["position"]
            edge_distances = [ship_pos[0], map_size[0]-ship_pos[0], ship_pos[1], map_size[1]-ship_pos[1]]
            # min_edge_distance = min(edge_distances)
            
            ship_vel = ship_state["velocity"]
            # ship_speed = math.sqrt(ship_vel[0]**2 + ship_vel[1]**2)

            mine_conditions = (
                len(close_threats) >= 5 and               # multiple threats
                not self.recent_mine_dropped()            # not too frequent
            )

            if mine_conditions:
                self.last_mine_time = self.eval_frames
                return True
        
            return False
        except Exception as e:
            print(e)
            return False

    def recent_mine_dropped(self):
        """Check if mine was dropped recently to prevent spamming"""
        if self.last_mine_time + 120 >= self.eval_frames:
            self.last_mine_time =  self.eval_frames
            return False
        else:
            return True


    def assess_threats(self, asteroids, ship_state, mines):
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

        for mine in mines:
            distance = math.dist(ship_pos, mine['position']) 
            
            # threat score based on distance
            threat_score = 1.0 / (distance + 0.01)
            threat_score = min(1.0, threat_score)
            
            # calculate approach speed 
            to_mine = [mine['position'][0]-ship_pos[0], mine['position'][1]-ship_pos[1]] # vector from ship to asteroid
            distance_norm = math.sqrt(to_mine[0]**2 + to_mine[1]**2) # magnitude
            
            threats.append({
                'asteroid': mine,
                'distance': distance,
                'approach_speed': mine.remaining_time,
                'threat_score': threat_score
            })
        
        # print(threats)
        return threats

    def calculate_intercept(self, ship_pos, asteroid):
        """
        calculate intercept point and time using law of cosines approach
        return intercept position and time to intercept
        """
        asteroid_pos = asteroid['position']
        asteroid_vel = asteroid['velocity']
        bullet_speed = 850
        
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
            fire = 0

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
from kesslergame import KesslerController 
from typing import Dict, Tuple
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np
import pygad

class MyController(KesslerController):
    def __init__(self, chromosome=None):
        super().__init__()
        self.eval_frames = 0
        self.generation = 0
        self.performance_score = 0
        self.last_actions = []
        
        # Genetic parameters - will be evolved
        if chromosome is not None:
            self.chromosome = chromosome
            self.set_parameters_from_chromosome()
        else:
            self.chromosome = self.get_default_chromosome()
            self.set_parameters_from_chromosome()
        
        # Build fuzzy systems using genetic parameters
        self.build_targeting_system()
        self.build_movement_system()
        self.build_survival_system()
        
        # Genetic algorithm setup
        self.setup_genetic_algorithm()
    
    def get_default_chromosome(self):
        """Return default chromosome values for genetic evolution"""
        return [
            # Targeting parameters
            0.05,   # bullet_time_instant
            0.8,    # bullet_time_fast
            1.5,    # bullet_time_slow
            0.1,    # aim_error_excellent
            0.4,    # aim_error_good
            0.6,    # aim_error_poor
            0.5,    # fire_threshold
            
            # Movement parameters
            200.0,  # threat_safe
            400.0,  # threat_caution_start
            600.0,  # threat_caution_end
            700.0,  # threat_danger
            50.0,   # edge_at_edge
            100.0,  # edge_near_start
            220.0,  # edge_near_end
            200.0,  # edge_safe
            
            # Survival parameters
            0.4,    # threat_low
            0.7,    # threat_medium
            0.3,    # health_critical
            0.3,    # survival_evade_threshold
            
            # Behavior weights
            0.6,    # aggression_level
            0.7,    # accuracy_priority
            0.4,    # risk_tolerance
        ]
    
    def set_parameters_from_chromosome(self):
        """Extract parameters from chromosome for fuzzy systems with validation"""
        # Extract raw values
        raw_params = {
            # Targeting
            'bullet_time_instant': self.chromosome[0],
            'bullet_time_fast': self.chromosome[1],
            'bullet_time_slow': self.chromosome[2],
            'aim_error_excellent': self.chromosome[3],
            'aim_error_good': self.chromosome[4],
            'aim_error_poor': self.chromosome[5],
            'fire_threshold': self.chromosome[6],
            
            # Movement
            'threat_safe': self.chromosome[7],
            'threat_caution_start': self.chromosome[8],
            'threat_caution_end': self.chromosome[9],
            'threat_danger': self.chromosome[10],
            'edge_at_edge': self.chromosome[11],
            'edge_near_start': self.chromosome[12],
            'edge_near_end': self.chromosome[13],
            'edge_safe': self.chromosome[14],
            
            # Survival
            'threat_low': self.chromosome[15],
            'threat_medium': self.chromosome[16],
            'health_critical': self.chromosome[17],
            'survival_evade_threshold': self.chromosome[18],
            
            # Behavior
            'aggression_level': self.chromosome[19],
            'accuracy_priority': self.chromosome[20],
            'risk_tolerance': self.chromosome[21],
        }
        
        # Apply constraints and bounds
        self.genetic_params = {}
        
        # Targeting parameters with ordering constraints
        bt_instant = max(0.01, raw_params['bullet_time_instant'])
        bt_fast = max(bt_instant + 0.1, raw_params['bullet_time_fast'])
        bt_slow = max(bt_fast + 0.1, raw_params['bullet_time_slow'])
        
        self.genetic_params['bullet_time_instant'] = bt_instant
        self.genetic_params['bullet_time_fast'] = bt_fast
        self.genetic_params['bullet_time_slow'] = bt_slow
        
        # Aim error parameters with ordering constraints
        aim_excellent = max(0.05, min(0.3, raw_params['aim_error_excellent']))
        aim_good = max(aim_excellent + 0.05, min(0.6, raw_params['aim_error_good']))
        aim_poor = max(aim_good + 0.05, min(0.8, raw_params['aim_error_poor']))
        
        self.genetic_params['aim_error_excellent'] = aim_excellent
        self.genetic_params['aim_error_good'] = aim_good
        self.genetic_params['aim_error_poor'] = aim_poor
        
        self.genetic_params['fire_threshold'] = max(0.3, min(0.8, raw_params['fire_threshold']))
        
        # Movement parameters with ordering constraints
        threat_safe = max(50, raw_params['threat_safe'])
        threat_caution_start = max(threat_safe + 50, raw_params['threat_caution_start'])
        threat_caution_end = max(threat_caution_start + 50, raw_params['threat_caution_end'])
        threat_danger = max(threat_caution_end + 50, raw_params['threat_danger'])
        
        self.genetic_params['threat_safe'] = threat_safe
        self.genetic_params['threat_caution_start'] = threat_caution_start
        self.genetic_params['threat_caution_end'] = threat_caution_end
        self.genetic_params['threat_danger'] = threat_danger
        
        # Edge parameters with ordering constraints
        edge_at_edge = max(10, raw_params['edge_at_edge'])
        edge_near_start = max(edge_at_edge + 20, raw_params['edge_near_start'])
        edge_near_end = max(edge_near_start + 30, raw_params['edge_near_end'])
        edge_safe = max(edge_near_end + 50, raw_params['edge_safe'])
        
        self.genetic_params['edge_at_edge'] = edge_at_edge
        self.genetic_params['edge_near_start'] = edge_near_start
        self.genetic_params['edge_near_end'] = edge_near_end
        self.genetic_params['edge_safe'] = edge_safe
        
        # Survival parameters
        self.genetic_params['threat_low'] = max(0.1, min(0.6, raw_params['threat_low']))
        self.genetic_params['threat_medium'] = max(self.genetic_params['threat_low'] + 0.1, min(0.9, raw_params['threat_medium']))
        self.genetic_params['health_critical'] = max(0.1, min(0.5, raw_params['health_critical']))
        self.genetic_params['survival_evade_threshold'] = max(0.1, min(0.5, raw_params['survival_evade_threshold']))
        
        # Behavior weights
        self.genetic_params['aggression_level'] = max(0.1, min(1.0, raw_params['aggression_level']))
        self.genetic_params['accuracy_priority'] = max(0.1, min(1.0, raw_params['accuracy_priority']))
        self.genetic_params['risk_tolerance'] = max(0.1, min(1.0, raw_params['risk_tolerance']))
    
    def setup_genetic_algorithm(self):
        """Setup genetic algorithm for evolution"""
        self.gene_space = [
            # Targeting parameters
            {'low': 0.01, 'high': 0.1},
            {'low': 0.2, 'high': 1.5},
            {'low': 0.5, 'high': 3.0},
            {'low': 0.05, 'high': 0.3},
            {'low': 0.1, 'high': 0.6},
            {'low': 0.2, 'high': 0.8},
            {'low': 0.3, 'high': 0.8},
            
            # Movement parameters
            {'low': 50, 'high': 300},
            {'low': 100, 'high': 500},
            {'low': 200, 'high': 700},
            {'low': 300, 'high': 1000},
            {'low': 10, 'high': 100},
            {'low': 30, 'high': 150},
            {'low': 50, 'high': 250},
            {'low': 100, 'high': 300},
            
            # Survival parameters
            {'low': 0.1, 'high': 0.6},
            {'low': 0.2, 'high': 0.9},
            {'low': 0.1, 'high': 0.5},
            {'low': 0.1, 'high': 0.5},
            
            # Behavior weights
            {'low': 0.1, 'high': 1.0},
            {'low': 0.1, 'high': 1.0},
            {'low': 0.1, 'high': 1.0},
        ]
    
    def build_targeting_system(self):
        """Targeting system that ensures consistent shooting"""
        bullet_time = ctrl.Antecedent(np.arange(0, 3.0, 0.01), 'bullet_time')
        aim_error = ctrl.Antecedent(np.arange(0, math.pi, 0.05), 'aim_error')
        asteroid_size = ctrl.Antecedent(np.arange(1, 5, 1), 'asteroid_size')
        
        turn_rate = ctrl.Consequent(np.arange(-180, 180, 1), 'turn_rate')
        fire_cmd = ctrl.Consequent(np.arange(0, 1, 0.1), 'fire_cmd')
        
        # GENETIC membership functions
        bt_instant = self.genetic_params['bullet_time_instant']
        bt_fast = self.genetic_params['bullet_time_fast']
        bt_slow = self.genetic_params['bullet_time_slow']
        
        bullet_time['instant'] = fuzz.trimf(bullet_time.universe, [0, 0, bt_instant])
        bullet_time['fast'] = fuzz.trimf(bullet_time.universe, [bt_instant, bt_fast, bt_slow])
        bullet_time['slow'] = fuzz.smf(bullet_time.universe, bt_slow, 3.0)
        
        aim_excellent = self.genetic_params['aim_error_excellent']
        aim_good = self.genetic_params['aim_error_good']
        aim_poor = self.genetic_params['aim_error_poor']
        
        aim_error['excellent'] = fuzz.trimf(aim_error.universe, [0, 0, aim_excellent])
        aim_error['good'] = fuzz.trimf(aim_error.universe, [aim_excellent, aim_good, aim_poor])
        aim_error['poor'] = fuzz.smf(aim_error.universe, aim_poor, math.pi)
        
        asteroid_size['small'] = fuzz.trimf(asteroid_size.universe, [1, 1, 2])
        asteroid_size['medium'] = fuzz.trimf(asteroid_size.universe, [1.5, 2.5, 3.5])
        asteroid_size['large'] = fuzz.trimf(asteroid_size.universe, [3, 4, 4])
        
        turn_rate['sharp_left'] = fuzz.trimf(turn_rate.universe, [-180, -180, -90])
        turn_rate['left'] = fuzz.trimf(turn_rate.universe, [-120, -60, 0])
        turn_rate['fine_tune'] = fuzz.trimf(turn_rate.universe, [-30, 0, 30])
        turn_rate['right'] = fuzz.trimf(turn_rate.universe, [0, 60, 120])
        turn_rate['sharp_right'] = fuzz.trimf(turn_rate.universe, [90, 180, 180])
        
        fire_threshold = self.genetic_params['fire_threshold']
        fire_cmd['no'] = fuzz.trimf(fire_cmd.universe, [0, 0, fire_threshold])
        fire_cmd['yes'] = fuzz.trimf(fire_cmd.universe, [fire_threshold, 1, 1])
        
        # AGGRESSIVE targeting rules - ensures consistent shooting
        aggression = self.genetic_params['aggression_level']
        accuracy = self.genetic_params['accuracy_priority']
        
        rules = []
        
        # High aggression: shoot very frequently
        if aggression > 0.7:
            rules.extend([
                ctrl.Rule(aim_error['excellent'] | aim_error['good'], fire_cmd['yes']),
                ctrl.Rule(asteroid_size['large'] & aim_error['poor'], fire_cmd['yes']),
                ctrl.Rule(bullet_time['instant'] | bullet_time['fast'], fire_cmd['yes']),
            ])
        # Medium aggression: balanced shooting
        elif aggression > 0.4:
            rules.extend([
                ctrl.Rule(aim_error['excellent'], fire_cmd['yes']),
                ctrl.Rule(aim_error['good'] & bullet_time['fast'], fire_cmd['yes']),
                ctrl.Rule(asteroid_size['large'] & aim_error['good'], fire_cmd['yes']),
            ])
        # Low aggression: conservative shooting
        else:
            rules.extend([
                ctrl.Rule(aim_error['excellent'] & bullet_time['instant'], fire_cmd['yes']),
                ctrl.Rule(aim_error['excellent'] & bullet_time['fast'], fire_cmd['yes']),
            ])
        
        # Always include these basic firing rules
        rules.extend([
            ctrl.Rule(aim_error['excellent'] & bullet_time['instant'], fire_cmd['yes']),
            ctrl.Rule(asteroid_size['large'] & aim_error['good'], fire_cmd['yes']),
        ])
        
        # Turn rate rules based on accuracy priority
        if accuracy > 0.7:
            rules.extend([
                ctrl.Rule(aim_error['excellent'], turn_rate['fine_tune']),
                ctrl.Rule(aim_error['good'], turn_rate['left']),
                ctrl.Rule(aim_error['good'], turn_rate['right']),
            ])
        else:
            rules.extend([
                ctrl.Rule(aim_error['excellent'], turn_rate['left']),
                ctrl.Rule(aim_error['excellent'], turn_rate['right']),
                ctrl.Rule(aim_error['good'], turn_rate['left']),
                ctrl.Rule(aim_error['good'], turn_rate['right']),
            ])
        
        # Basic turn rules
        rules.extend([
            ctrl.Rule(aim_error['poor'], turn_rate['sharp_left']),
            ctrl.Rule(aim_error['poor'], turn_rate['sharp_right']),
            ctrl.Rule(aim_error['poor'], fire_cmd['no']),
        ])
        
        self.targeting_control = ctrl.ControlSystem(rules)
        self.targeting_sim = ctrl.ControlSystemSimulation(self.targeting_control)
    
    def build_movement_system(self):
        """Optimized movement system for better threat response"""
        closest_threat_dist = ctrl.Antecedent(np.arange(0, 1000, 10), 'closest_threat_dist')
        threat_approach_speed = ctrl.Antecedent(np.arange(-500, 500, 10), 'threat_approach_speed')
        edge_distance = ctrl.Antecedent(np.arange(0, 500, 10), 'edge_distance')
        threat_density = ctrl.Antecedent(np.arange(0, 1, 0.1), 'threat_density')
        
        thrust_cmd = ctrl.Consequent(np.arange(-480, 480, 10), 'thrust_cmd')
        
        # GENETIC membership functions
        threat_safe = self.genetic_params['threat_safe']
        threat_caution = self.genetic_params['threat_caution_start']
        threat_danger = self.genetic_params['threat_danger']
        
        closest_threat_dist['safe'] = fuzz.trapmf(closest_threat_dist.universe, [0, 0, threat_safe, threat_caution])
        closest_threat_dist['caution'] = fuzz.trimf(closest_threat_dist.universe, [threat_safe, threat_caution, threat_danger])
        closest_threat_dist['danger'] = fuzz.smf(closest_threat_dist.universe, threat_danger, 1000)
        
        threat_approach_speed['moving_away'] = fuzz.trapmf(threat_approach_speed.universe, [-500, -500, -150, 0])
        threat_approach_speed['stable'] = fuzz.trimf(threat_approach_speed.universe, [-100, 0, 100])
        threat_approach_speed['approaching_fast'] = fuzz.smf(threat_approach_speed.universe, 80, 500)
        
        edge_at_edge = self.genetic_params['edge_at_edge']
        edge_near = self.genetic_params['edge_near_start']
        edge_safe = self.genetic_params['edge_safe']
        
        edge_distance['at_edge'] = fuzz.trapmf(edge_distance.universe, [0, 0, edge_at_edge, edge_near])
        edge_distance['near_edge'] = fuzz.trimf(edge_distance.universe, [edge_at_edge, edge_near, edge_safe])
        edge_distance['safe_center'] = fuzz.smf(edge_distance.universe, edge_safe, 500)
        
        threat_low = self.genetic_params['threat_low']
        threat_medium = self.genetic_params['threat_medium']
        
        threat_density['low'] = fuzz.trimf(threat_density.universe, [0, 0, threat_low])
        threat_density['medium'] = fuzz.trimf(threat_density.universe, [threat_low, threat_medium, 0.8])
        threat_density['high'] = fuzz.smf(threat_density.universe, 0.7, 1.0)
        
        thrust_cmd['full_reverse'] = fuzz.trimf(thrust_cmd.universe, [-480, -480, -240])
        thrust_cmd['reverse'] = fuzz.trimf(thrust_cmd.universe, [-360, -180, 0])
        thrust_cmd['stop'] = fuzz.trimf(thrust_cmd.universe, [-60, 0, 60])
        thrust_cmd['forward'] = fuzz.trimf(thrust_cmd.universe, [0, 180, 360])
        thrust_cmd['full_forward'] = fuzz.trimf(thrust_cmd.universe, [240, 480, 480])
        
        # OPTIMIZED movement rules based on threat level
        rules = []
        
        # HIGH THREAT RESPONSE: Evade immediate dangers
        rules.extend([
            # Immediate danger: very close AND fast approaching
            ctrl.Rule(closest_threat_dist['danger'] & threat_approach_speed['approaching_fast'], thrust_cmd['full_reverse']),
            
            # Multiple threats closing in
            ctrl.Rule(threat_density['high'] & closest_threat_dist['caution'] & threat_approach_speed['approaching_fast'], thrust_cmd['reverse']),
            
            # Single threat in danger zone
            ctrl.Rule(closest_threat_dist['danger'] & threat_approach_speed['stable'], thrust_cmd['reverse']),
        ])
        
        # MEDIUM THREAT RESPONSE: Strategic positioning
        rules.extend([
            # Maintain distance from medium threats
            ctrl.Rule(closest_threat_dist['caution'] & threat_approach_speed['approaching_fast'], thrust_cmd['stop']),
            
            # Reposition when threatened but not in immediate danger
            ctrl.Rule(threat_density['medium'] & closest_threat_dist['caution'], thrust_cmd['forward']),
        ])
        
        # LOW THREAT & POSITIONING
        rules.extend([
            # Edge avoidance - highest priority after threats
            ctrl.Rule(edge_distance['at_edge'], thrust_cmd['full_forward']),
            ctrl.Rule(edge_distance['near_edge'], thrust_cmd['forward']),
            
            # Safe movement for better firing angles
            ctrl.Rule(closest_threat_dist['safe'] & threat_density['low'], thrust_cmd['forward']),
            
            # Stop when in ideal position
            ctrl.Rule(closest_threat_dist['safe'] & threat_density['low'] & edge_distance['safe_center'], thrust_cmd['stop']),
            
            # Default forward movement when no immediate concerns
            ctrl.Rule(threat_approach_speed['moving_away'] & threat_density['low'], thrust_cmd['forward']),
        ])
        
        self.movement_control = ctrl.ControlSystem(rules)
        self.movement_sim = ctrl.ControlSystemSimulation(self.movement_control)

    def build_survival_system(self):
        """Genetic survival system"""
        threat_level = ctrl.Antecedent(np.arange(0, 1, 0.1), 'threat_level')
        health_status = ctrl.Antecedent(np.arange(0, 1, 0.1), 'health_status')
        survival_override = ctrl.Consequent(np.arange(0, 1, 0.1), 'survival_override')
        
        # GENETIC membership functions
        threat_low = self.genetic_params['threat_low']
        threat_medium = self.genetic_params['threat_medium']
        health_critical = self.genetic_params['health_critical']
        
        threat_level['low'] = fuzz.trimf(threat_level.universe, [0, 0, threat_low])
        threat_level['medium'] = fuzz.trimf(threat_level.universe, [threat_low, threat_medium, 1.0])
        threat_level['high'] = fuzz.trimf(threat_level.universe, [threat_medium, 1, 1])
        
        health_status['critical'] = fuzz.trimf(health_status.universe, [0, 0, health_critical])
        health_status['healthy'] = fuzz.trimf(health_status.universe, [health_critical, 1, 1])
        
        survival_override['normal'] = fuzz.trimf(survival_override.universe, [0, 0, 0.3])
        survival_override['evade'] = fuzz.trimf(survival_override.universe, [0.2, 0.5, 0.8])
        survival_override['panic'] = fuzz.trimf(survival_override.universe, [0.7, 1, 1])
        
        rules = [
            ctrl.Rule(threat_level['high'] | health_status['critical'], survival_override['panic']),
            ctrl.Rule(threat_level['medium'] & health_status['critical'], survival_override['panic']),
            ctrl.Rule(threat_level['medium'] & health_status['healthy'], survival_override['evade']),
            ctrl.Rule(threat_level['low'] & health_status['healthy'], survival_override['normal']),
        ]
        
        self.survival_control = ctrl.ControlSystem(rules)
        self.survival_sim = ctrl.ControlSystemSimulation(self.survival_control)

    def calculate_intercept(self, ship_pos, ship_heading, asteroid):
        """Calculate intercept point and time with moving target"""
        asteroid_pos = asteroid.position
        asteroid_vel = asteroid.velocity
        bullet_speed = 800
        
        dx = asteroid_pos[0] - ship_pos[0]
        dy = asteroid_pos[1] - ship_pos[1]
        
        a = asteroid_vel[0]**2 + asteroid_vel[1]**2 - bullet_speed**2
        b = 2 * (dx * asteroid_vel[0] + dy * asteroid_vel[1])
        c = dx**2 + dy**2
        
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return None, float('inf')
            
        t1 = (-b + math.sqrt(discriminant)) / (2*a)
        t2 = (-b - math.sqrt(discriminant)) / (2*a)
        
        intercept_time = min(t for t in [t1, t2] if t > 0) if any(t > 0 for t in [t1, t2]) else max(t1, t2)
        
        if intercept_time <= 0:
            return None, float('inf')
            
        intercept_x = asteroid_pos[0] + asteroid_vel[0] * intercept_time
        intercept_y = asteroid_pos[1] + asteroid_vel[1] * intercept_time
        
        return (intercept_x, intercept_y), intercept_time

    def assess_threats(self, asteroids, ship_state, game_state):
        """Enhanced threat assessment with better prioritization"""
        ship_pos = ship_state["position"]
        ship_vel = ship_state["velocity"]
        
        threats = []
        
        for asteroid in asteroids:
            distance = math.dist(ship_pos, asteroid.position)
            
            relative_vel = [
                asteroid.velocity[0] - ship_vel[0],
                asteroid.velocity[1] - ship_vel[1]
            ]
            
            to_asteroid = [
                asteroid.position[0] - ship_pos[0],
                asteroid.position[1] - ship_pos[1]
            ]
            
            distance_norm = math.sqrt(to_asteroid[0]**2 + to_asteroid[1]**2)
            if distance_norm > 0:
                approach_speed = (relative_vel[0] * to_asteroid[0] + relative_vel[1] * to_asteroid[1]) / distance_norm
            else:
                approach_speed = 0
            
            time_to_collision = distance / max(abs(approach_speed), 1) if approach_speed > 0 else float('inf')
            
            # Enhanced threat scoring with multiple factors
            distance_factor = 1.0 / (distance + 0.1)
            speed_factor = abs(approach_speed) / 200.0  # Normalize by max expected speed
            size_factor = asteroid.size / 4.0  # Normalize by max size
            collision_factor = 1.0 / (time_to_collision + 0.1)
            
            # Combined threat score (0-1 range)
            threat_score = min(1.0, 0.4 * distance_factor + 0.3 * speed_factor + 0.2 * size_factor + 0.1 * collision_factor)
            
            # Calculate intercept opportunity
            ship_heading = math.radians(ship_state["heading"])
            intercept_pos, intercept_time = self.calculate_intercept(ship_pos, ship_heading, asteroid)
            if intercept_time == float('inf'):
                shoot_opportunity = 0.0
            else:
                # Higher opportunity for faster intercept and better aim
                dx = intercept_pos[0] - ship_pos[0]
                dy = intercept_pos[1] - ship_pos[1]
                target_angle = math.atan2(dy, dx)
                aim_error = abs(target_angle - ship_heading)
                aim_error = min(aim_error, 2*math.pi - aim_error)
                
                time_opportunity = 1.0 / (intercept_time + 0.1)
                aim_opportunity = 1.0 / (aim_error + 0.1)
                shoot_opportunity = min(1.0, 0.6 * time_opportunity + 0.4 * aim_opportunity)
            
            threats.append({
                'asteroid': asteroid,
                'distance': distance,
                'approach_speed': approach_speed,
                'time_to_collision': time_to_collision,
                'threat_score': threat_score,
                'shoot_opportunity': shoot_opportunity,
                'size': asteroid.size
            })
        
        return threats

    def find_best_target(self, threats, ship_state):
        """Find best target considering both threat level and shooting opportunity"""
        ship_pos = ship_state["position"]
        ship_heading = math.radians(ship_state["heading"])
        
        best_target = None
        best_score = -float('inf')
        best_intercept = None
        best_time = float('inf')
        
        for threat in threats:
            asteroid = threat['asteroid']
            
            intercept_pos, intercept_time = self.calculate_intercept(ship_pos, ship_heading, asteroid)
            
            if intercept_time == float('inf'):
                continue
            
            # Calculate aiming information
            dx = intercept_pos[0] - ship_pos[0]
            dy = intercept_pos[1] - ship_pos[1]
            target_angle = math.atan2(dy, dx)
            aim_error = abs(target_angle - ship_heading)
            aim_error = min(aim_error, 2*math.pi - aim_error)
            
            # Enhanced scoring: balance between threat elimination and shooting efficiency
            threat_elimination_score = threat['threat_score'] * 2.0  # Higher weight for threats
            shooting_efficiency_score = threat['shoot_opportunity'] * 1.5
            size_bonus = threat['size'] * 0.5  # Prefer larger asteroids
            
            score = threat_elimination_score + shooting_efficiency_score + size_bonus
            
            if score > best_score:
                best_score = score
                best_target = asteroid
                best_intercept = intercept_pos
                best_time = intercept_time
                
        return best_target, best_intercept, best_time

    def actions(self, ship_state: Dict, game_state) -> Tuple[float, float, bool, bool]:
        """Optimized main controller with better threat response"""
        asteroids = game_state.asteroids
        if not asteroids:
            return (0.0, 0.0, False, False)
        
        ship_pos = ship_state["position"]
        ship_heading = math.radians(ship_state["heading"])
        ship_vel = ship_state["velocity"]
        
        # Enhanced threat assessment
        threats = self.assess_threats(asteroids, ship_state, game_state)
        
        if not threats:
            return (0.0, 0.0, False, False)
        
        # Find best target using enhanced threat assessment
        best_asteroid, intercept_pos, intercept_time = self.find_best_target(threats, ship_state)
        
        if not best_asteroid:
            return (0.0, 0.0, False, False)
        
        # Calculate aiming information
        dx = intercept_pos[0] - ship_pos[0]
        dy = intercept_pos[1] - ship_pos[1]
        target_angle = math.atan2(dy, dx)
        aim_error = abs(target_angle - ship_heading)
        aim_error = min(aim_error, 2*math.pi - aim_error)
        
        # Calculate threat metrics
        closest_threat = min(threats, key=lambda x: x['distance'])
        threat_count = len([t for t in threats if t['distance'] < 400])
        threat_density = threat_count / len(asteroids) if asteroids else 0
        
        # Calculate overall threat level (max threat score)
        max_threat_level = max([t['threat_score'] for t in threats]) if threats else 0
        
        # Calculate edge safety
        map_size = game_state.map_size
        edge_distances = [
            ship_pos[0], map_size[0] - ship_pos[0],
            ship_pos[1], map_size[1] - ship_pos[1]
        ]
        min_edge_distance = min(edge_distances)
        
        # GENETIC FUZZY TARGETING with threat awareness
        try:
            self.targeting_sim.input['bullet_time'] = min(intercept_time, 3.0)
            self.targeting_sim.input['aim_error'] = aim_error
            self.targeting_sim.input['asteroid_size'] = best_asteroid.size
            self.targeting_sim.compute()
            turn_rate = float(self.targeting_sim.output['turn_rate'])
            fire_output = self.targeting_sim.output['fire_cmd']
            fire = bool(fire_output > 0.5)
        except Exception as e:
            # Smart fallback: adjust firing based on threat level
            turn_rate = float(aim_error * 180 / math.pi)
            # Fire more aggressively under high threat
            fire_threshold = 0.2 if max_threat_level > 0.7 else 0.1
            fire = bool(aim_error < fire_threshold and intercept_time < 2.0)
        
        # GENETIC FUZZY MOVEMENT - Threat-based decisions
        thrust = 50.0  # Conservative default movement
        
        if closest_threat:
            try:
                self.movement_sim.input['closest_threat_dist'] = closest_threat['distance']
                self.movement_sim.input['threat_approach_speed'] = closest_threat['approach_speed']
                self.movement_sim.input['edge_distance'] = min_edge_distance
                self.movement_sim.input['threat_density'] = threat_density
                self.movement_sim.compute()
                thrust = float(self.movement_sim.output['thrust_cmd'])
            except Exception as e:
                # Threat-based fallback movement
                if closest_threat['distance'] < 100 and closest_threat['approach_speed'] > 150:
                    thrust = -300.0  # Strong reverse for immediate danger
                elif closest_threat['distance'] < 200:
                    thrust = -100.0  # Gentle reverse for medium danger
                elif min_edge_distance < 50:
                    thrust = 200.0  # Move away from edges
                else:
                    thrust = 80.0  # Slow forward for positioning
        
        # GENETIC SURVIVAL OVERRIDE - Smart panic response
        try:
            health = ship_state["lives_remaining"] / 3.0
            self.survival_sim.input['threat_level'] = max_threat_level  # Use max threat level
            self.survival_sim.input['health_status'] = health
            self.survival_sim.compute()
            survival_mode = self.survival_sim.output['survival_override']
            
            # Graduated survival response
            if survival_mode > 0.8:  # Panic mode
                thrust = -400.0
                # Only shoot if perfectly aimed in panic
                fire = fire and (aim_error < 0.1)
            elif survival_mode > 0.5:  # Evade mode
                thrust = -200.0
                # Reduced shooting in evade mode
                fire = fire and (aim_error < 0.15)
        except Exception as e:
            pass
        
        # STRATEGIC MINE DEPLOYMENT - Based on threat analysis
        drop_mine = False
        
        # Only consider mines if we have multiple close threats
        close_threats = [t for t in threats if t['distance'] < 250 and t['approach_speed'] > 50]
        if len(close_threats) >= 2:  # Multiple threats approaching
            # Don't drop mines at edges
            if min_edge_distance > 120:
                # Check if we're moving away from threats (good time for mines)
                avg_approach_speed = sum(t['approach_speed'] for t in close_threats) / len(close_threats)
                if thrust < 0 or avg_approach_speed > 80:  # Moving away or fast threats
                    drop_mine = True
        
        self.eval_frames += 1
        
        # Performance monitoring
        if self.eval_frames % 120 == 0:  # Every 2 seconds
            avg_threat = sum(t['threat_score'] for t in threats) / len(threats) if threats else 0
            print(f"Frame {self.eval_frames}: Thrust={thrust:.1f}, Turn={turn_rate:.1f}, Fire={fire}, Mine={drop_mine}")
            print(f"  Threats: {len(close_threats)} close, AvgThreat={avg_threat:.2f}, MaxThreat={max_threat_level:.2f}")
        
        return (float(thrust), float(turn_rate), bool(fire), bool(drop_mine))

    def evolve(self, population_size=12, generations=10):
        """Evolve the controller using genetic algorithm"""
        def fitness_func(ga_instance, solution, solution_idx):
            test_controller = MyController(chromosome=solution)
            fitness = self.evaluate_chromosome(test_controller)
            return fitness
        
        initial_population = [self.chromosome]
        for _ in range(population_size - 1):
            initial_population.append(self.generate_random_chromosome())
        
        self.ga_instance = pygad.GA(
            num_generations=generations,
            num_parents_mating=population_size // 2,
            fitness_func=fitness_func,
            sol_per_pop=population_size,
            num_genes=len(self.gene_space),
            gene_space=self.gene_space,
            parent_selection_type="tournament",
            keep_parents=1,
            crossover_type="uniform",
            mutation_type="random",
            mutation_percent_genes=15,
            initial_population=initial_population
        )
        
        self.ga_instance.run()
        
        best_solution, best_fitness, _ = self.ga_instance.best_solution()
        self.chromosome = best_solution
        self.set_parameters_from_chromosome()
        self.generation += 1
        self.performance_score = best_fitness
        
        print(f"Generation {self.generation} completed. Best fitness: {best_fitness:.3f}")
        return best_fitness
    
    def generate_random_chromosome(self):
        """Generate a random chromosome"""
        chromosome = []
        for gene_space in self.gene_space:
            low = gene_space['low']
            high = gene_space['high']
            chromosome.append(np.random.uniform(low, high))
        return chromosome
    
    def evaluate_chromosome(self, controller):
        """Evaluate a chromosome's fitness"""
        fitness = 1.0
        
        # Reward good parameter relationships
        if (controller.genetic_params['aim_error_excellent'] < 
            controller.genetic_params['aim_error_good'] < 
            controller.genetic_params['aim_error_poor']):
            fitness += 2.0
        
        if (controller.genetic_params['threat_safe'] < 
            controller.genetic_params['threat_caution_start'] < 
            controller.genetic_params['threat_caution_end'] < 
            controller.genetic_params['threat_danger']):
            fitness += 2.0
        
        # Reward balanced behavior
        aggression = controller.genetic_params['aggression_level']
        accuracy = controller.genetic_params['accuracy_priority']
        risk = controller.genetic_params['risk_tolerance']
        
        balance_penalty = abs(aggression - 0.6) + abs(accuracy - 0.7) + abs(risk - 0.4)
        fitness += (1.0 - balance_penalty)
        
        return max(0.1, fitness)

    @property
    def name(self) -> str:
        return f"My Controller v{self.generation}"
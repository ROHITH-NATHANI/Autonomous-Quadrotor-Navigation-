#! /usr/bin/env python

import sys
import os
import argparse
import pandas as pd
import numpy as np
import time
import math
try:
    import rospy
    from std_msgs.msg import Empty
    from geometry_msgs.msg import Twist, Pose
    HAS_ROS = True
except ImportError:
    HAS_ROS = False

ACTIONS     = ['up', 'down', 'left', 'right']
LENGTH      = 5
N_STATES    = LENGTH * LENGTH
START       = (LENGTH - 1, 0)   # (4, 0) => World (0.0, 0.0) [Red Start Pad]
TERMINAL    = (0, 4)            # (0, 4) => World (8.0, -8.0) [Green Goal Pillar]

# 6 Non-linear, scattered random obstacles across the 5x5 grid
HOLE1       = (0, 1)            # World (8.0, -2.0)
HOLE2       = (1, 3)            # World (6.0, -6.0)
HOLE3       = (2, 1)            # World (4.0, -2.0)
HOLE4       = (2, 4)            # World (4.0, -8.0)
HOLE5       = (3, 2)            # World (2.0, -4.0)
HOLE6       = (4, 3)            # World (0.0, -6.0)

# Default Training Hyperparameters
DEFAULT_EPISODES = 1000
GAMMA            = 0.9       # Discount factor
ALPHA            = 0.1       # Learning rate
EPSILON_START    = 1.0       # 100% random path exploration at start
EPSILON_MIN      = 0.05      # Minimum exploration rate
EPSILON_DECAY    = 0.005     # Exponential decay per episode

SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
Q_TABLE_FILE     = os.path.join(SCRIPT_DIR, 'q_table.csv')

def build_q_table():
    return pd.DataFrame(
        np.zeros((N_STATES, len(ACTIONS))),
        columns=ACTIONS
    )

def is_hole(state):
    return state in (HOLE1, HOLE2, HOLE3, HOLE4, HOLE5, HOLE6)

def get_env_feedback(state, action):
    """
    Reward function that shapes optimal, shortest-path navigation:
      - Goal reached:        +50.0
      - Obstacle collision:  -50.0
      - Wall collision:       -5.0
      - Step penalty:         -1.0 (forces minimum steps / shortest path)
      - Distance shaping:     +2.0 when closing distance to goal, -2.0 when straying
    """
    a, b = state
    prev_dist = abs(a - TERMINAL[0]) + abs(b - TERMINAL[1])

    if action == 'up':
        a = max(0, a - 1)
    elif action == 'down':
        a = min(LENGTH - 1, a + 1)
    elif action == 'left':
        b = max(0, b - 1)
    elif action == 'right':
        b = min(LENGTH - 1, b + 1)
    
    next_state = (a, b)
    curr_dist = abs(a - TERMINAL[0]) + abs(b - TERMINAL[1])
    
    if next_state == TERMINAL:
        reward = 50.0
        end = True
    elif is_hole(next_state):
        reward = -50.0
        end = True
    elif next_state == state:
        reward = -5.0   # Boundary wall collision
        end = False
    else:
        # Step penalty (-1.0) + distance shaping
        reward = -1.0 + (prev_dist - curr_dist) * 2.0
        end = False

    return next_state, reward, end

def actor(observation, q_table, epsilon=0.0):
    """
    Epsilon-greedy policy:
      - Explore random paths with probability epsilon
      - Exploit the highest Q-value action otherwise
    """
    if np.random.uniform() < epsilon:
        return np.random.choice(ACTIONS)
    else:
        state_action = q_table.loc[observation, :]
        max_val = np.max(state_action)
        best_actions = state_action[state_action == max_val].index
        return np.random.choice(best_actions)

def train_agent(num_episodes=DEFAULT_EPISODES, save_path=Q_TABLE_FILE):
    """
    Trains the Q-learning agent through random path exploration, gradually
    annealing epsilon to discover the shortest and most optimized path.
    """
    q_table = build_q_table()
    epsilon = EPSILON_START
    
    print("\n" + "=" * 70)
    print(" TRAINING Q-LEARNING AGENT (Random Path Exploration -> Optimal)")
    print(" Episodes: %d | Alpha: %.2f | Gamma: %.2f | Epsilon: 1.0 -> 0.05" % 
          (num_episodes, ALPHA, GAMMA))
    print("=" * 70)

    for episode in range(num_episodes):
        state = START
        end = False
        step = 0
        episode_reward = 0.0

        while not end and step < 50:
            a, b = state
            act = actor(a * LENGTH + b, q_table, epsilon=epsilon)
            next_state, reward, end = get_env_feedback(state, act)
            na, nb = next_state
            episode_reward += reward
            
            q_predict = q_table.loc[a * LENGTH + b, act]
            if next_state != TERMINAL:
                q_target = reward + GAMMA * q_table.iloc[na * LENGTH + nb].max()
            else:
                q_target = reward
            q_table.loc[a * LENGTH + b, act] += ALPHA * (q_target - q_predict)
            state = next_state
            step += 1

        # Exponential decay of exploration rate
        epsilon = max(EPSILON_MIN, epsilon * (1.0 - EPSILON_DECAY))

        # Periodic logging
        if (episode + 1) % 100 == 0 or episode == 0:
            status = "EXPLORING" if epsilon > 0.3 else "CONVERGING" if epsilon > 0.1 else "OPTIMIZING"
            print("Episode %4d/%d | Reward: %+6.1f | Steps: %2d | Randomness (Eps): %.3f | Phase: %s" % 
                  (episode + 1, num_episodes, episode_reward, step, epsilon, status))

    print("=" * 70)
    print(" Training Finished Successfully!")
    
    # Save learned Q-table
    try:
        q_table.to_csv(save_path, index=True)
        print(" Learned Q-Table saved to: %s" % save_path)
    except Exception as e:
        print(" Warning: Could not save Q-table to file: %s" % e)
        
    print("=" * 70)
    return q_table

def load_or_train_q_table(num_episodes=DEFAULT_EPISODES, save_path=Q_TABLE_FILE, force_train=False):
    if not force_train and os.path.exists(save_path):
        print("\nLoading pre-trained Q-table from: %s" % save_path)
        q_table = pd.read_csv(save_path, index_col=0)
        # Ensure column types and names match
        q_table.columns = ACTIONS
        return q_table
    return train_agent(num_episodes=num_episodes, save_path=save_path)

def evaluate_best_path(q_table):
    """
    Extracts the best learned optimal path (100% greedy, 0% randomness).
    """
    maze_transitions = []
    states_visited = [START]
    state = START
    end = False
    step = 0
    total_reward = 0.0

    print("\n--- Testing Learned Optimal Shortest Path ---")
    while not end and step < 20:
        a, b = state
        act = actor(a * LENGTH + b, q_table, epsilon=0.0)  # Pure greedy
        next_state, reward, end = get_env_feedback(state, act)
        total_reward += reward
        print("Step %d: State %s -> Action: %-5s -> Next: %s | Step Reward: %+5.1f | Total: %5.1f" %
              (step + 1, state, act, next_state, reward, total_reward))
        maze_transitions.append(act)
        state = next_state
        states_visited.append(state)
        step += 1

    print("\n==> Selected Optimal Path: %d steps | Cumulative Reward: %.1f <==" % 
          (len(maze_transitions), total_reward))
    return maze_transitions, states_visited, total_reward

def print_maze_overview(states_visited):
    visited_set = set(states_visited)
    print("\n5x5 Maze Visual Map (*=Goal, S=Start, .=Path, X=Obstacle):")
    for r in range(LENGTH):
        row_str = "  "
        for c in range(LENGTH):
            pos = (r, c)
            if pos == TERMINAL:
                row_str += " * "
            elif pos == START:
                row_str += " S "
            elif is_hole(pos):
                row_str += " X "
            elif pos in visited_set:
                row_str += " o "
            else:
                row_str += " . "
        print(row_str)
    print()

def droneActions(maze_transitions):
    dir_map = {'up': 0, 'down': 1, 'right': 2, 'left': 3}
    return [dir_map[action] for action in maze_transitions]

def droneMotions(actions):
    """
    Computes relative drone maneuvers based on heading:
      0  = straight forward
      1  = 90 deg turn left, then forward
     -1  = 90 deg turn right, then forward
      2  = 180 deg U-turn, then forward
    """
    TURN_MAP = {
        (0, 0): 0,  (0, 1): 2,  (0, 2): -1, (0, 3): 1,
        (1, 0): 2,  (1, 1): 0,  (1, 2): 1,  (1, 3): -1,
        (2, 0): 1,  (2, 1): -1, (2, 2): 0,  (2, 3): 2,
        (3, 0): -1, (3, 1): 1,  (3, 2): 2,  (3, 3): 0,
    }

    current_heading = 0 # Drone begins facing +X ('up')
    drone_move = []
    for target_dir in actions:
        move = TURN_MAP[(current_heading, target_dir)]
        drone_move.append(move)
        current_heading = target_dir

    return drone_move

###################################### CLOSED-LOOP DRONE CONTROLLER #####################################

class MoveDroneClass(object):

    def __init__(self):
        self.ctrl_c = False
        self.rate = rospy.Rate(10)
        self._pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self._pub_takeoff = rospy.Publisher('/drone/takeoff', Empty, queue_size=1)
        self._takeoff_msg = Empty()
        self._pub_land = rospy.Publisher('/drone/land', Empty, queue_size=1)
        self._land_msg = Empty()

        # Real-time ground truth pose from sjtu_drone plugin
        self.cur_x = 0.0
        self.cur_y = 0.0
        self.cur_z = 0.0
        self.cur_yaw = 0.0
        self.has_pose = False
        self._sub_pose = rospy.Subscriber('/drone/gt_pose', Pose, self._pose_callback)

    def _pose_callback(self, msg):
        self.cur_x = msg.position.x
        self.cur_y = msg.position.y
        self.cur_z = msg.position.z
        self.cur_yaw = 2.0 * math.atan2(msg.orientation.z, msg.orientation.w)
        self.has_pose = True

    def stop_drone(self):
        cmd = Twist()
        self._pub_cmd_vel.publish(cmd)

    def fly_to_waypoint(self, tx, ty, timeout=14.0):
        start_time = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            if not self.has_pose:
                self.rate.sleep()
                continue

            dx = tx - self.cur_x
            dy = ty - self.cur_y
            dist = math.hypot(dx, dy)

            # Reached waypoint target
            if dist < 0.25:
                rospy.loginfo("Reached Waypoint (%.2f, %.2f) | Error: %.3fm" % (tx, ty, dist))
                self.stop_drone()
                break

            if rospy.Time.now().to_sec() - start_time > timeout:
                rospy.logwarn("Waypoint (%.2f, %.2f) timed out; continuing." % (tx, ty))
                self.stop_drone()
                break

            target_yaw = math.atan2(dy, dx)
            yaw_err = math.atan2(math.sin(target_yaw - self.cur_yaw), math.cos(target_yaw - self.cur_yaw))

            cmd = Twist()
            if abs(yaw_err) > 0.20:
                cmd.linear.x = 0.0
                cmd.angular.z = max(-1.0, min(1.0, 2.5 * yaw_err))
            else:
                cmd.linear.x = max(0.25, min(0.65, 1.2 * dist))
                cmd.angular.z = max(-0.5, min(0.5, 1.5 * yaw_err))

            self._pub_cmd_vel.publish(cmd)
            self.rate.sleep()

        time.sleep(0.3)

    def execute_path(self, states_visited):
        waypoints = []
        for (r, c) in states_visited[1:]:
            wx = (4 - r) * 2.0
            wy = -c * 2.0
            waypoints.append((wx, wy, r, c))

        # =================== DRONE TAKEOFF ===================
        rospy.loginfo("Taking off from Start pad...")
        for _ in range(3):
            self._pub_takeoff.publish(self._takeoff_msg)
            time.sleep(1.0)
        time.sleep(2.5)
        rospy.loginfo("Drone airborne! Current pos: (%.2f, %.2f, %.2f)" % (self.cur_x, self.cur_y, self.cur_z))

        # =================== EXECUTE OPTIMIZED WAYPOINTS ===================
        total = len(waypoints)
        for idx, (tx, ty, gr, gc) in enumerate(waypoints):
            rospy.loginfo(">>> Navigating to Step %d/%d: Grid(%d,%d) -> World(%.1f, %.1f) <<<" % 
                          (idx + 1, total, gr, gc, tx, ty))
            self.fly_to_waypoint(tx, ty)

        # =================== GOAL REACHED & LAND ===================
        rospy.loginfo("=== GOAL REACHED! LANDING DRONE AT GREEN GOAL ===")
        self.stop_drone()
        time.sleep(1.0)
        for _ in range(3):
            self._pub_land.publish(self._land_msg)
            rospy.loginfo("Landing...")
            time.sleep(1.0)
        rospy.loginfo("Mission Accomplished! Final Drone pos: (%.2f, %.2f)" % (self.cur_x, self.cur_y))

def main():
    parser = argparse.ArgumentParser(description="Reinforcement Learning Q-Learning Drone Navigation")
    parser.add_argument('--train', nargs='?', const=DEFAULT_EPISODES, type=int, default=None,
                        help="Train the Q-learning agent for N episodes (default: 1000) and save Q-table.")
    parser.add_argument('--fly', action='store_true',
                        help="Fly the drone in Gazebo using the saved/best learned policy.")
    parser.add_argument('--episodes', type=int, default=DEFAULT_EPISODES,
                        help="Number of training episodes (default: 1000).")
    args = parser.parse_args()

    # Determine mode:
    # 1. --train specified: run training only and save Q-table
    # 2. --fly specified: load saved Q-table and fly
    # 3. No flags: train (or load) and fly immediately
    if args.train is not None:
        episodes = args.train if isinstance(args.train, int) else args.episodes
        q_table = train_agent(num_episodes=episodes, save_path=Q_TABLE_FILE)
        transitions, states, reward = evaluate_best_path(q_table)
        print_maze_overview(states)
        print("\nTraining complete! To fly the drone with this trained policy, run:")
        print("  rosrun fly_drone drone_fly_qlearn.py --fly\n")
        return

    # If --fly or default:
    force_train = (not args.fly) and (not os.path.exists(Q_TABLE_FILE))
    q_table = load_or_train_q_table(num_episodes=args.episodes, save_path=Q_TABLE_FILE, force_train=force_train)
    
    print("\n=== Learned Q-Table ===")
    print(q_table)
    print("========================\n")

    maze_transitions, states_visited, total_reward = evaluate_best_path(q_table)
    print_maze_overview(states_visited)

    actions = droneActions(maze_transitions)
    print("Actions sequence    :", actions)
    drone_motions = droneMotions(actions)
    print("Relative drone move :", drone_motions)
    print("Total moves to fly  :", len(drone_motions))

    # Initialize ROS node and fly drone
    if not HAS_ROS:
        print("\n[NOTE] ROS (rospy) is not loaded in this environment.")
        print("To fly the drone in Gazebo, run inside your ROS workspace:")
        print("  rosrun fly_drone drone_fly_qlearn.py --fly\n")
        return

    rospy.init_node('move_drone')
    drone_ctrl = MoveDroneClass()
    try:
        drone_ctrl.execute_path(states_visited)
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()

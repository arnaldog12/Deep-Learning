import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6 # 6 == len(pose)
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        x, y, z, a_x, a_y, a_z = self.sim.pose
        v_x, v_y, v_z = self.sim.v
        ang_vx, ang_vy, ang_vz = self.sim.angular_v
        
        reward = np.tanh(1 - 0.003*(abs(self.sim.pose[:3] - self.target_pos))).sum()
#         reward = 1.0 - 0.003*(abs(self.sim.pose[:3] - self.target_pos)).sum() # reach a target
#         reward += 0.002*z # takeoff
        return reward
        
        '''
        x, y, z, a_x, a_y, a_z = self.sim.pose
        v_x, v_y, v_z = self.sim.v
        ang_vx, ang_vy, ang_vz = self.sim.angular_v
        
        reward = np.sum(np.abs([x, y, z] - self.target_pos)) # reach a target
#         reward -= 0.005*v_z # land softly
#         reward += 0.002*z**2 # takeoff
#         reward -= 0.0005*((ang_vx**2) + (ang_vy**2) + (ang_vz**2)) # hover in place
#         reward += 0.02 # fly as long as possible
        return reward
        '''

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
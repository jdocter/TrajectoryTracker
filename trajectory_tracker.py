# assuming a pose is global x, y
# 3-12 meters between updats? assuming 5-20 Hz updats and 150mph
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import seaborn as sns
from scipy.stats import multivariate_normal

norm = np.linalg.norm


class Agent: # maybe want to create a new id for each agent? one per car
  """
    Represents a CarAgent. State is described as
      global position (x, y), velocity (vx, vy), acceleration? angular acceleration?
      can we get colors of car? ==> better probability distribution

    Claim: Future pose is markovian. That is only current state is predictive of the future states.
    We don't need to save poses from previous timesteps to predict trajectories. Intuitively,
    we don't expect other teams to plan trajectories based on past states, because we don't either
    (and we are the smartest :)). Competetitors' trajectories are likely planned based on only on
    current (x,y, vx, vy, a, etc) and environment variables, so we can predict their trajectoreis
    from the same information.
  """
  def __init__(self, pose, t, vx = 5, vy = 5, ax = 0, ay = 0, trajectory_length = 20):
    # initial velocities and accelration guesses?
    # vx, vy is the velocity orthogonal and along the vector implied by phi
    self.vx = vx
    self.vy = vy
    # list of [x, y, vx, vy, t]'s
    self.trajectory = [[pose[0]-0.1, pose[1]-0.2, vx, vy, t],[pose[0], pose[1], vx, vy, t]]
    self.trajectory_length = trajectory_length

  def likelihood(self, x, y, t):
    """
      return the probability that at time t, this CarAgent ends up in Pose pose
      represent pose at time t as a bivariate guassian distribution with mean
      x + vx*t, y + vy+t, and variance ~ vx, vy (for now just have constant variance)

    """
    # first transform pose into fram of car using phi
    # then calcualte probability using bivariate guassian distribution

    trajectory = np.array(self.trajectory)
    xs = trajectory[:,0]
    ys = trajectory[:,1]
    vxs = trajectory[:,2]
    vys = trajectory[:,3]
    ts = trajectory[:,4]
    delta_xs = xs[1:] - (xs[:-1] + vxs[:-1]*(ts[1:] - ts[:-1]))
    delta_ys = ys[1:] - (ys[:-1] + vys[:-1]*(ts[1:] - ts[:-1]))
    print(ys[1:])
    print(delta_ys)

    x_i, y_i, vx, vy, t_i = self.trajectory[-1]
    prior_x = x_i + vx*(t - t_i)
    prior_y = y_i + vy*(t - t_i)
    print(np.stack([xs,ys]))
    cov = np.cov(np.stack([xs, ys]))
    print(cov)
    # heuristic for covariance
    prior = multivariate_normal(mean=[prior_x, prior_y],
                                cov=[[5,0],[0,5]])
    return prior.pdf([x,y])

  def likelihoods(self, poses, t):
    """

    """
    trajectory = np.array(self.trajectory)
    xs = trajectory[:,0]
    ys = trajectory[:,1]
    vxs = trajectory[:,2]
    vys = trajectory[:,3]
    ts = trajectory[:,4]
    delta_xs = xs[1:] - (x[:-1] + vxs*(ts[1:] - ts[:-1]))
    delta_ys = ys[1:] - (y[:-1] + vys*(ts[1:] - ts[:-1]))



    x_i, y_i, vx, vy, t_i = trajectory[-1]
    prior_x = x + vx*(t - t_i)
    prior_y = y + vy*(t - t_i)
    poses_in_frame = [self.frame_of(x_i,y_i,self.vx,self.vy,x_f,y_f) for x_f,y_f in poses]
    # heuristic for covariance
    multivariate_normal(poses_in_frame, cov = [[norm(self.vx*t,self.vy*t),0],[0,norm(self.vx*t,self.vy*t)]])

  def frame_of(x_i,y_i,vx,vy,x_f,y_f):
    """
      transform new pose (x_f,y_f) into frame of last pose + velocity
    """
    delta = [x_f - x_i, y_f - y_i]
    frame_y = [vx, vy]
    frame_x = [vy, -vx]
    return np.dot(delta,frame_x)/norm(frame_x), np.dot(delta,frame_y)/norm(frame_x)


  def update(self, pose):
    """
     update state based on new pose
    """
    self.trajectory.append[pose]
    if len(self.trajectory) > self.trajectory_length:
      self.trajectory = self.trajectory[len(self.trajectory) - self.trajectory_length:]
    self.vx = trajectory[-1][2]
    self.vy = trajectory[-1][3]


 # Global State: Set of CarAgents
# given new set of poses, calculate prob(pose|caragent) for each pose and each caragent
# use MLE to pick poses and update car agents
# need some thresshold value to determine when to create new car agent

# get probability out of field of view ...


class Tracker:
  def __init__(self):
    self.agents = set() # want to create a number for each agent, by index.

  def add_agent(self, new_agent): # i.e. become aware of a new agent
    self.agents.add(new_agent)

  def remove_agemt(self, untrackable_agent): # i.e. remove an agent we are no longer keeping track of (if the agent has not appeared in the last K observations)
    self.agents.remove(untrackable_agent)

  def update(self, poses, t):
    # Assigns an agent to each pose. Does this greedily and removes agents from a set.
    # Returns a dictionary (to be modified) of the agent and its new pose

    estimated_poses = {}
    agents_list = self.agents.copy()

    full_probabilities = {}

    if len(poses) == len(agents_list):

      for x,y in poses:
        likelihoods = [agent.likelihood(x, y, t) for agent in agents_list] # a list of agent probabilities
        full_probabilities[(x,y)] = likelihoods # need full information of all pose probabilities
        # [p(agent 0), p agent1, p agent 2, ... , p agent n]
      print(full_probabilities)
      for x,y in full_probabilities.keys():
        # find the optimum probability and remove it: (find index of the maximum probability)
        #found = False
        #while(not found):
        max_agent = full_probabilities[(x,y)].index(max(full_probabilities[x,y]))
          #if max_agent in agents_list:
        agents_list.remove(max_agent)
        estimated_poses[max_agent] = (x,y)
            #found = True

            # want to set probability of all other occurences of this car to 0
        for x,y in full_probabilities:
          full_probabilities[(x,y)][max_agent] = 0

    elif len(poses) < len(agents_list):
      # figure out a pose for all existing cars
      # assign the remaining pose as a starting pose for the new agent
      pass
    else: # have more poses than agents, create a new agent

      # getting rangom poses
      self.add_agent(Agent(next(iter(poses)), t))
    return estimated_poses


def get_distance(p1, p2):
  return ((p1[0] -p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

class TrajectorySim:
  def __init__(self, dt, velocity, traj, p_step_size):
    self.p = 0 # parametric parameter
    self.t = 0
    self.dt = dt
    self.p_step_size = p_step_size
    self.velocity = velocity
    self.traj = traj

  def step(self):
    goal_dist = self.velocity(self.t)
    dist = 0
    while dist < goal_dist:
      dist += get_distance(self.traj(self.p + self.p_step_size), self.traj(self.p))
      self.p += self.p_step_size
    self.t += self.dt
    return self.traj(self.p)

class Simulator:
  def __init__(self, trajectory_sim = None):
    ellipse = lambda p : (1000*np.cos(2*np.pi*p/1000), 500 * np.sin(2*np.pi*p/1000))
    velocity = lambda t : 60 + 5*np.sin(2*np.pi*t/5)
    self.dt = 0.05
    self.trajectory_sim = TrajectorySim(0.05,velocity, ellipse, 1) if trajectory_sim is None else trajectory_sim

  def simulate(self, tracker, total_time, dt):
    t = 0
    truth = {'x':[], 'y': []}
    while t < total_time:
      t += dt
      data = set([self.trajectory_sim.step(),])
      tracker.update(data, t)
      print(data)
      print(type(data))
      d = data.pop()
      truth['x'].append(d[0])
      truth['y'].append(d[1])
    # snsplot = sns.lineplot(truth)
    # snsplot.savefig('truth_trajectory.png')

    for agent in tracker.agents:
      trajectory = np.array(self.trajectory)
      xs = trajectory[:,0]
      ys = trajectory[:,0]
      self.plot_trajectory(xs, ys, truth['x'], truth['y'])

  def plot_trajectory(self, xs, ys, truth_xs, truth_ys):
   plt.xlabel("meters")
   plt.ylabel("meters")
   plt.title("Trajectory")
   plt.plot(xs, ys)
   # plt.plot(truth_xs, truth_ys)
   plt.savefig("trajectory.png")


if __name__=='__main__':
    sim = Simulator()
    tracker = Tracker()
    sim.simulate(tracker,40,0.5)
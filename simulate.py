# assuming a pose is global x, y
# 3-12 meters between updats? assuming 5-20 Hz updats and 150mph
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import seaborn as sns
from scipy.stats import multivariate_normal

norm = np.linalg.norm

from trajectory_tracker import tracker

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
import numpy as np

STEP_GAIN = 1.0 / 0.05


def p_action(eef, goal, grip, max_a=1.0, gain=0.6):
  a = np.zeros(7)
  a[:3] = np.clip((np.asarray(goal) - np.asarray(eef)) * STEP_GAIN * gain, -max_a, max_a)
  a[6] = grip
  return a


class Script:
  def __init__(self, target_xy, rim=0.055, grasp_dz=0.02, lift=0.10, place_dz=0.025, noise=0.0, rng=None,
      obj="akita_black_bowl_1"):
    self.t = np.asarray(target_xy, float)
    self.obj = obj
    self.rim, self.grasp_dz, self.lift, self.place_dz = rim, grasp_dz, lift, place_dz
    self.noise = noise
    self.rng = rng or np.random.default_rng(0)
    self.phase, self.count = 0, 0
    self.offset = np.array([0.0, rim])

  def __call__(self, obs):
    eef = obs["robot0_eef_pos"]
    bowl = obs[self.obj + "_pos"]
    z_table = 0.90
    above = 0.12
    ph = self.phase
    self.count += 1
    if ph == 0:
      goal = np.r_[bowl[:2] + self.offset, z_table + above]
      a = p_action(eef, goal, -1.0)
      if np.linalg.norm(goal - eef) < 0.01 or self.count > 80:
        self.phase, self.count = 1, 0
    elif ph == 1:
      goal = np.r_[bowl[:2] + self.offset, bowl[2] + self.grasp_dz]
      a = p_action(eef, goal, -1.0, gain=0.5)
      if np.linalg.norm(goal - eef) < 0.006 or self.count > 60:
        self.phase, self.count = 2, 0
    elif ph == 2:
      a = p_action(eef, eef, 1.0)
      if self.count > 15:
        self.phase, self.count = 3, 0
    elif ph == 3:
      self.hold = eef[:2] - bowl[:2]
      goal = np.r_[eef[:2], z_table + self.lift + 0.05]
      a = p_action(eef, goal, 1.0)
      if eef[2] > z_table + self.lift or self.count > 50:
        self.phase, self.count = 4, 0
    elif ph == 4:
      goal = np.r_[self.t + (eef[:2] - bowl[:2]), z_table + self.lift + 0.05]
      a = p_action(eef, goal, 1.0)
      if np.linalg.norm((bowl[:2] - self.t)) < 0.006 or self.count > 120:
        self.phase, self.count = 5, 0
    elif ph == 5:
      goal = np.r_[self.t + (eef[:2] - bowl[:2]), z_table + self.place_dz + (eef[2] - bowl[2])]
      a = p_action(eef, goal, 1.0, gain=0.5)
      if bowl[2] < z_table + 0.005 or self.count > 60:
        self.phase, self.count = 6, 0
    elif ph == 6:
      a = p_action(eef, eef, -1.0)
      if self.count > 12:
        self.phase, self.count = 7, 0
    else:
      a = p_action(eef, np.r_[eef[:2], z_table + 0.20], -1.0)
      if self.count > 25:
        self.phase = 8
    if self.noise:
      a[:3] = np.clip(a[:3] + self.rng.normal(0.0, self.noise, 3), -1, 1)
    return a

  @property
  def done(self):
    return self.phase >= 8


def for_vessel(entry, target_xy, rim_in=0.010, **kw):
  r = 0.5 * min(entry["size_x"], entry["size_y"])
  return Script(target_xy, rim=r - rim_in, grasp_dz=max(0.02, entry["height"] - 0.035), obj=entry["key"] + "_1", **kw)


def success(obs, target_xy, radius=0.03, obj="akita_black_bowl_1"):
  bowl = obs[obj + "_pos"]
  return bool(np.linalg.norm(bowl[:2] - np.asarray(target_xy)) < radius and bowl[2] < 0.92)

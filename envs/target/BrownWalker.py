import numpy as np
from typing import Optional, Union

class BrownWalker:
    def __init__(self, init_pose=np.array([80.0, 80.0, 0.0]), step_size: float = 1.0,
                    map_shape=None, sigma_x: float = 0.8, sigma_y: float = 1.2,
                    sigma_theta: float = 0.0, rng: Optional[Union[int, np.random.Generator]] = None,):
        
        self.pose = np.array(init_pose, dtype=np.float32)
        self.step_size = step_size
        self.map_shape = map_shape
        self.sigma_x, self.sigma_y, self.sigma_theta = sigma_x, sigma_y, sigma_theta

        if isinstance(rng, np.random.Generator):
            self.rng = rng
        elif rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(int(rng))

        self._t = 0

    def step(self):
        noise = self.rng.normal(
            loc=0.0,
            scale=[self.sigma_x, self.sigma_y, self.sigma_theta],
            size=(3,),
        )
        self.pose += noise.astype(np.float32)

        # theta wrap
        self.pose[2] = (self.pose[2] + np.pi) % (2 * np.pi) - np.pi

        if self.map_shape is not None:
            h, w = self.map_shape
            self.pose[0] = np.clip(self.pose[0], 0, w - 1)
            self.pose[1] = np.clip(self.pose[1], 0, h - 1)

        self._t += 1

    def get_pose(self):
        return self.pose.copy()
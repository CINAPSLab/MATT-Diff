import numpy as np
import math
import random
from typing import Optional, Sequence, Tuple

def line_free(grid: np.ndarray, p0: Tuple[float, float], p1: Tuple[float, float]) -> bool:
    H, W = grid.shape
    x0, y0 = map(int, map(round, p0))
    x1, y1 = map(int, map(round, p1))

    dx = abs(x1 - x0); sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0); sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        if not (0 <= x < W and 0 <= y < H):
            return False
        if grid[y, x] == 0:
            return False
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return True

def path_is_free(grid: np.ndarray, path: Optional[np.ndarray]) -> bool:
    if path is None or len(path) < 2:
        return False
    for a, b in zip(path[:-1], path[1:]):
        if not line_free(grid, (a[0], a[1]), (b[0], b[1])):
            return False
    return True

class Node:
    __slots__ = ("x", "y", "parent", "cost")
    def __init__(self, x: float, y: float,
                 parent: Optional["Node"] = None, cost: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.parent = parent
        self.cost = float(cost)

class RRTStar:
    def __init__(self, start: Sequence[float], goal: Sequence[float], grid: np.ndarray,
                 max_iter: int = 2000, step_size: float = 20.0, goal_sample_rate: float = 0.30,
                 search_radius: float = 35.0, goal_radius: float = 12.0):
        self.start = Node(*start)
        self.goal = Node(*goal)
        self.grid = grid
        self.H, self.W = grid.shape
        self.max_iter = int(max_iter)
        self.step_size = float(step_size)
        self.goal_sample_rate = float(goal_sample_rate)
        self.search_radius = float(search_radius)
        self.goal_radius = float(goal_radius)
        self.nodes = [self.start]

    def plan(self) -> Optional[np.ndarray]:
        for _ in range(self.max_iter):
            rnd = self._sample()
            near = self._nearest(rnd)
            new = self._steer(near, rnd)

            if not line_free(self.grid, (near.x, near.y), (new.x, new.y)):
                continue

            near_idx = self._near_indices(new, self.search_radius)
            parent = near
            min_cost = near.cost + self._dist(near, new)
            for j in near_idx:
                nb = self.nodes[j]
                if line_free(self.grid, (nb.x, nb.y), (new.x, new.y)):
                    c = nb.cost + self._dist(nb, new)
                    if c < min_cost:
                        parent, min_cost = nb, c
            new.parent, new.cost = parent, min_cost
            self.nodes.append(new)

            for j in near_idx:
                nb = self.nodes[j]
                if nb is parent:
                    continue
                if line_free(self.grid, (new.x, new.y), (nb.x, nb.y)):
                    c = new.cost + self._dist(new, nb)
                    if c + 1e-6 < nb.cost:
                        nb.parent, nb.cost = new, c

            if self._dist(new, self.goal) <= self.goal_radius:
                if line_free(self.grid, (new.x, new.y), (self.goal.x, self.goal.y)):
                    final = Node(self.goal.x, self.goal.y,
                                 parent=new, cost=new.cost + self._dist(new, self.goal))
                    return self._extract(final)

        best = min(self.nodes, key=lambda n: self._dist(n, self.goal))
        return self._extract(best)

    def _sample(self) -> Node:
        if random.random() < self.goal_sample_rate:
            return self.goal
        return Node(random.uniform(0, self.W - 1), random.uniform(0, self.H - 1))

    def _nearest(self, target: Node) -> Node:
        return min(self.nodes, key=lambda n: self._dist(n, target))

    def _near_indices(self, node: Node, r: float) -> list:
        return [i for i, n in enumerate(self.nodes) if self._dist(n, node) <= r]

    def _steer(self, a: Node, b: Node) -> Node:
        dx, dy = (b.x - a.x), (b.y - a.y)
        d = math.hypot(dx, dy)
        if d <= self.step_size:
            return Node(b.x, b.y, parent=a, cost=a.cost + d)
        r = self.step_size / (d + 1e-9)
        return Node(a.x + r * dx, a.y + r * dy, parent=a, cost=a.cost + self.step_size)

    @staticmethod
    def _dist(n1: Node, n2: Node) -> float:
        return math.hypot(n1.x - n2.x, n1.y - n2.y)

    def _extract(self, node: Node) -> np.ndarray:
        path = []
        cur = node
        while cur is not None:
            path.append([cur.x, cur.y])
            cur = cur.parent
        path.reverse()
        if abs(path[-1][0] - self.goal.x) > 1e-3 or abs(path[-1][1] - self.goal.y) > 1e-3:
            path.append([self.goal.x, self.goal.y])
        return np.asarray(path, dtype=np.float32)
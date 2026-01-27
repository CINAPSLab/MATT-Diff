import numpy as np
import cv2

def display2map(map_origin, ratio, x_d):
    
    d2m = np.array([[0, 1 / ratio, 0],
                    [-1 / ratio, 0, 0],
                    [0, 0, 1]])
    return d2m @ (x_d - map_origin)

def map2display(map_origin, ratio, x_m):
    m2d = np.array([[0, -ratio, 0],
                    [ratio, 0, 0],
                    [0, 0, 1]])
    return m2d @ x_m + map_origin

def SE2_kinematics(x, action, tau):
    wt_2 = action[1] * tau / 2
    t_v_sinc_term = tau * action[0] * np.sinc(wt_2 / np.pi)
    ret_x = np.empty(3)
    ret_x[0] = x[0] + t_v_sinc_term * np.cos(x[2] + wt_2)
    ret_x[1] = x[1] + t_v_sinc_term * np.sin(x[2] + wt_2)
    ret_x[2] = x[2] + 2 * wt_2
    return ret_x


def DDA(grid, x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0
    steps = int(max(abs(dx), abs(dy)))
    if steps == 0:
        return int(round(x0)), int(round(y0))
    xinc, yinc = dx/steps, dy/steps
    x, y = float(x0), float(y0)
    H, W = grid.shape
    for _ in range(steps):
        col = int(round(x)); row = int(round(y))
        if 0 <= row < H and 0 <= col < W:
            if grid[row, col] == 0:  # 0=障害物
                break
        else:
            break
        x += xinc
        y += yinc
    return int(round(x)), int(round(y))

def observation_model(self, tgt_pose):
    dx = tgt_pose[0] - self._rbt[0]
    dy = tgt_pose[1] - self._rbt[1]
    r  = np.hypot(dx, dy) + np.random.normal(0, self.range_noise)
    bearing = (np.arctan2(dy, dx) - self._rbt[2]
               + np.random.normal(0, self.bearing_noise))
    return np.array([r, bearing])

def SDF_RT(robot_pose, fov, radius, RT_res, grid, inner_r=10):
    pts = raytracing(grid, robot_pose, fov, radius, RT_res)
    x0, y0, theta = robot_pose
    x1i = x0 + inner_r * np.cos(theta - 0.5 * fov)
    y1i = y0 + inner_r * np.sin(theta - 0.5 * fov)
    x2i = x0 + inner_r * np.cos(theta + 0.5 * fov)
    y2i = y0 + inner_r * np.sin(theta + 0.5 * fov)
    pts = [[x1i, y1i]] + pts + [[x2i, y2i], [x1i, y1i]]
    return vertices_filter(np.array(pts, dtype=np.float32))


def raytracing(grid, robot_pose, fov, radius, RT_res):
    x0, y0, theta = robot_pose
    x1 = x0 + radius * np.cos(theta - 0.5 * fov)
    y1 = y0 + radius * np.sin(theta - 0.5 * fov)
    x2 = x0 + radius * np.cos(theta + 0.5 * fov)
    y2 = y0 + radius * np.sin(theta + 0.5 * fov)
    xs = np.linspace(x1, x2, RT_res)
    ys = np.linspace(y1, y2, RT_res)
    pts = []
    for xm, ym in zip(xs, ys):
        xx, yy = DDA(grid, int(x0), int(y0), int(xm), int(ym))
        if not pts or (yy != pts[-1][1] or xx != pts[-1][0]):
            pts.append([xx, yy])
    return pts



def vertices_filter(polygon, angle_threshold=0.05):
    diff = polygon[1:] - polygon[:-1]
    n = np.linalg.norm(diff, axis=1) + 1e-12
    u = diff / n[:, None]
    cosang = np.einsum('ij,ij->i', u[:-1], u[1:])
    ang = np.abs(np.arccos(np.clip(cosang, -1.0, 1.0)))
    keep = [True] + list(ang > angle_threshold) + [True]
    return polygon[keep, :]

def polygon_SDF(polygon, point):
    N = len(polygon) - 1
    e = polygon[1:] - polygon[:-1]
    v = point - polygon[:-1]
    t = np.clip((v*e).sum(1) / (e*e).sum(1), 0, 1)
    pq = v - e * t[:, None]
    d = (pq*pq).sum(1).min()
    wn = 0
    for i in range(N):
        i2 = (i + 1) % N
        if (polygon[i,1] <= point[1] < polygon[i2,1]) and np.cross(e[i], v[i]) > 0: wn += 1
        if (polygon[i,1] >  point[1] >= polygon[i2,1]) and np.cross(e[i], v[i]) < 0: wn -= 1
    sign = -1 if wn != 0 else 1  # 内=-1（負）、外=+1（正）
    return np.sqrt(d) * sign


def load_houseexpo_image_as_grid(image_path,  threshold=127):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height,width = image.shape[:2]
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    image = cv2.resize(image, (width,height), interpolation=cv2.INTER_NEAREST)## resize with shape


    # binarize
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    obstacles = (binary == 0).astype(np.uint8)

    
    grid = np.where(obstacles == 1, 0, 1).astype(np.uint8)
    return grid




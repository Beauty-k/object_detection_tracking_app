import math
from math import radians, tan

fov_degree = 114
fov_rad = math.radians(fov_degree)
distance_mm = 900

# Width of visible area in mm using cone formula
scene_width_mm = 2 * distance_mm * math.tan(fov_rad / 2)

# Pixels per mm
frame_width_px = 848
pixel_per_mm = frame_width_px / scene_width_mm


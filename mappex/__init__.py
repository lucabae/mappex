"""Indoor mapping with a simulated drone: detection, depth and a 2D map.

The package is split into small modules:

``scene``       geometry constants and the MuJoCo scene path
``simulation``  the drone: physics, flight primitives and camera streams
``depth``       pinhole camera maths, pixels and depths to world points
``detector``    YOLO detection and weights resolution
``mapping``     the accumulated object map and its OpenCV rendering
``dataset``     synthetic dataset generation from the scene
``train``       dataset assembly and YOLO training
``pipeline``    the exploration loop that wires the modules together
"""

__version__ = "0.1.0"

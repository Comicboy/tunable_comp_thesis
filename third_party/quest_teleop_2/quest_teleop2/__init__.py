"""Quest 3 teleoperation of a UR5e and a Wuji Hand 2, and demonstration recording for ACT.

The right hand is tracked (wrist pose -> TCP, finger pose -> hand joints); a left-hand pinch
freezes the arm. With the tactile glove on the robot hand, the hand can follow the operator
through an admittance to the measured contact forces (quest_teleop2.control).
"""

__version__ = "0.1.0"

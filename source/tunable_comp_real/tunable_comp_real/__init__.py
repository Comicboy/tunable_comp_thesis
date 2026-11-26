"""Real-robot side of the thesis (UR5e + Wuji Hand 2 + tactile glove).

Builds on the supervisor's quest_teleop2 package (third_party/quest_teleop_2, used unchanged) and
imports nothing from Isaac Lab, so it runs in the `teleop` env. Modules:

    hand_model   Hand 2 kinematics from its URDF: FK, fingertip Jacobians, pad normals, distal COMs,
                 0..1 <-> rad
    admittance   the simulation's dynamic / static admittance laws (numpy) and the fingertip layer:
                 glove force -> offset e -> DLS joint offset -> joint command
    hand_driver  MeasuredWujiHand2: the supervisor's Hand 2 driver with state() returning the last
                 measurement (never the command) when no new frame has arrived
    tactile_admittance
                 TactileFingertipAdmittance: our hand_filter for his TeleopSession (glove ->
                 fingertip forces -> admittance -> command; recorded keys incl. per-tick history;
                 hidden K)
    episodes     episode logic of the recording: success check (grasped + lifted 15 cm for 3 s),
                 one recorder per K with delete-last, the hidden weighted K draw
    session      RealTeleopSession: his TeleopSession with our episode flow (home, pinch + release
                 start, automatic success, r / d / h keys, hidden K)
"""
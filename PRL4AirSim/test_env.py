from Start import run_command
from Utils import connectUE
import Utils
import time
import numpy as np

windowX = 0
windowY = 1000

run_command(
    'gnome-terminal -- bash -c "./Linux/{projectName}.sh -windowed -WinX={WinX} -WinY={WinY} -NoVSync'.format(
        projectName="PathingEnv", WinX=windowX, WinY=windowY
    )
)

time.sleep(5)

connectUE("127.0.0.1", 29001)

from Simulation import Sim  

s = Sim((32, 32), 1)

s.resetBatch()
print("Reset done!")
for _ in range(5000):
    currentPos = (
        Utils.getClient()
        .getMultirotorState(s.droneObjects[0].droneName)
        .kinematics_estimated.position.to_numpy_array()
    )
    # s.is_collision(s.droneObjects[0])

    goal = s.droneObjects[0].currentGoal

    action = (goal - currentPos) * 0.5    

    print("goal: ", goal)
    print("currentPos: ", currentPos)
    print("requested action: ", action)
    action[2] = 0
    s.droneObjects[0].currentAction = action
    
    # self.is_collision(droneObject)
    if s.is_collision(s.droneObjects[0]):
        print("aaaaa")
        s.resetBatch()
    
    s.doActionBatch()
    s.gatherAllObservations()

    drone_pos = (
        Utils.getClient()
        .getMultirotorState(s.droneObjects[0].droneName)
        .kinematics_estimated.position.to_numpy_array()
    )

    target_dist_curr = float(np.linalg.norm(s.droneObjects[0].currentGoal - drone_pos))
    if target_dist_curr < s.goal_threshold:
        print("done?????????")
        s.resetBatch()


    
    print("measured vel: ", s.droneObjects[0].currentState["velocity"])

import pdb; pdb.set_trace()
import time

import numpy as np

from msgpackrpc.future import Future
from typing import Optional
class DroneObject(object):
    def __init__(self, droneId):
        self.droneId = droneId
        self.droneName = "Drone{}".format(droneId)
        self.currentArena = None
        self.currentStep = 0
        self.droneSpawnOffset = np.array([0, 0 * droneId, 0])
        
        self.start_pos = None

        self.previous_depth_image = None

        self.currentState = None
        self.currentStatePos = None  # Used to create the value heat map
        self.previousState = None
        self.currentAction = None
        self.currentTotalReward = 0
        self.distanceFromGoal = None

        self.currentGoal = None

        self.reseting = True
        self.reseting_API = False
        self.reseting_API_2 = False

        self.resetTick = 0
        self.resetFuture : Optional[Future] = None
        self.resetingTime = time.perf_counter()

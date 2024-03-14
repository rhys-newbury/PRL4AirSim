import Utils as Utils
import airsim
import numpy as np
import time
import DroneObj as DroneObj
import random
import argparse
from os.path import exists
import os
import pathlib
import binvox_rw
from pathlib import Path
import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D

beforeTime = None
afterTime = None


class Sim(object):
    def __init__(self, image_shape, num_drones):
        self.image_shape = image_shape

        self.origin_UE = np.array([0.0, 0.0, 910.0])

        self.create_voxel_grid()
        binvox_path = Path.cwd() / "training_env.binvox"

        with open(binvox_path, "rb") as f:
            self.map = binvox_rw.read_as_3d_array(f)

        self.droneObjects = [DroneObj.DroneObject(i) for i in range(num_drones)]
        self.episodes = 0
        self.model_download_at_episode = 0
        self.numImagesSent = 0

        # TODO: HyperParameters
        self.step_length = 0.25
        self.constant_x_vel = 1.0
        self.constant_z_pos = Utils.convert_pos_UE_to_AS(
            origin_UE=self.origin_UE, pos_UE=[8600.0, -4160.0, 1510.0]
        )[2]
        self.actionTime = 1.0
        self.goal_threshold = 0.30  # TODO: add in config file
        self.resetBatch()

    def create_voxel_grid(self):
        output_path = Path.cwd() / "training_env.binvox"
        if output_path.exists():
            return

        client = Utils.getClient()  # This doesn't work
        # client = airsim.VehicleClient()
        center = airsim.Vector3r(0, 0, 0)
        voxel_size = 100
        res = 1

        client.simCreateVoxelGrid(
            center, voxel_size, voxel_size, voxel_size, res, str(output_path)
        )
        print("voxel map generated!")

        with open(output_path, "rb") as f:
            map = binvox_rw.read_as_3d_array(f)
        # Set every below ground level as "occupied". #TODO: add inflation to the map
        map.data[:, :, :50] = True
        map.data[:, :, 70:] = True
        # binvox_edited_path = os.path.join(os.getcwd(), "block_edited.binvox")
        binvox_edited_path = Path.cwd() / "training_env.binvox"

        with open(binvox_edited_path, "wb") as f:
            binvox_rw.write(map, f)

    def interpolate(self, start, end, t):
        """Linearly interpolate between start and end points."""
        return [start[i] + t * (end[i] - start[i]) for i in range(3)]

    def check_obstacle_between(self, start, end, occupancy_grid, debug=False):
        """
        Check if there is an obstacle between two points in a 3D grid.
        Uses a simplified line traversal algorithm.
        If debug is True, it also plots the current point for visualization.
        """

        # Function to plot the current point for debugging
        def plot_debug_point(start, end, current):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            # Start and end points
            ax.scatter(start[0], start[1], start[2], color="green", label="Start")
            ax.scatter(end[0], end[1], end[2], color="red", label="End")

            # Current point
            ax.scatter(
                current[0], current[1], current[2], color="blue", label="Current"
            )

            # Labels and legend
            ax.set_xlabel("X axis")
            ax.set_ylabel("Y axis")
            ax.set_zlabel("Z axis")
            ax.legend()

            plt.show()

        # Calculate total number of steps needed for the interpolation
        num_steps = int(
            np.linalg.norm(np.array(end) - np.array(start)) * 2
        )  # x2 for higher resolution

        for step in range(num_steps):
            t = step / float(num_steps)
            current = self.interpolate(start, end, t)
            grid_point = [int(round(coord)) for coord in current]

            # Check bounds
            if any(
                coord < 0 or coord >= occupancy_grid.shape[i]
                for i, coord in enumerate(grid_point)
            ):
                continue  # Skip points outside the grid

            # Check if current grid point is an obstacle
            if occupancy_grid[tuple(grid_point)] != 0:
                # Plot the current point for debugging
                if debug:
                    plot_debug_point(start, end, current)
                return True  # Obstacle found

        return False  # No obstacle found

    def sample_start_goal_pos(self):
        # find free space points
        free_space_points = np.argwhere(self.map.data == 0)

        min_distance = 15
        max_distance = 25

        iter = 0

        while True:
            iter += 1
            start_point = random.choice(free_space_points)
            end_point = random.choice(free_space_points)

            if (
                min_distance
                <= np.linalg.norm(np.array(start_point) - np.array(end_point))
                <= max_distance
            ):
                if self.check_obstacle_between(start_point, end_point, self.map.data):
                    start_pos = [
                        start_point[1] + self.map.translate[0],
                        start_point[0] + self.map.translate[1],
                        abs(self.map.translate[2]) - start_point[2],
                    ]
                    goal_pos = [
                        end_point[1] + self.map.translate[0],
                        end_point[0] + self.map.translate[1],
                        abs(self.map.translate[2]) - end_point[2],
                    ]

                    return start_pos, goal_pos

    def gatherAllObservations(self):
        # nonResetingDrones = []
        nonResetingDrones = [x for x in self.droneObjects if not x.reseting]
        # for droneObject in self.droneObjects:
        #     if not droneObject.reseting:
        #         nonResetingDrones.append(droneObject)

        if len(nonResetingDrones) == 0:
            return

        imageMessage = [
            airsim.ImageRequest(
                1,
                airsim.ImageType.DepthPlanar,
                True,
                True,
            )
            for _ in nonResetingDrones
        ]
        vehicleNames = [vehicle.droneName for vehicle in nonResetingDrones]

        beforeTime = time.perf_counter()
        responses_raw = Utils.getClient().client.call(
            "ANSR_simGetBatchImages", imageMessage, vehicleNames
        )
        afterTime = time.perf_counter()

        print("Gather images: ", afterTime - beforeTime)

        responses = [
            airsim.ImageResponse.from_msgpack(response_raw)
            for response_raw in responses_raw
        ]
        imageDepths = [
            airsim.list_to_2d_float_array(
                responses[i].image_data_float,
                responses[i].width,
                responses[i].height,
            )
            for i in range(len(responses))
        ]

        for i, droneObject in enumerate(nonResetingDrones):
            imageDepth = imageDepths[i]
            if imageDepth.size == 0:
                print("Image size is 0")
                imageDepth = (
                    np.ones(shape=(self.image_shape[1], self.image_shape[2])) * 30
                )

            maxDistance = 50
            imageDepth[imageDepth > maxDistance] = maxDistance
            imageDepth = imageDepth.astype(np.uint8)

            if droneObject.currentStep == 0:
                droneObject.previous_depth_image = imageDepth

            stacked_images = np.array([imageDepth, droneObject.previous_depth_image])

            multirotorState = Utils.getClient().getMultirotorState(
                droneObject.droneName
            )
            velocity = (
                multirotorState.kinematics_estimated.linear_velocity.to_numpy_array()
            )
            droneObject.previous_depth_image = imageDepth

            droneObject.previousState = droneObject.currentState
            droneObject.currentState = {
                "image": stacked_images,
                "velocity": velocity,
                "goal": droneObject.currentGoal,
            }
            droneObject.currentStatePos = (
                multirotorState.kinematics_estimated.position.to_numpy_array()
            )

    def doActionBatch(self):
        droneNames = []
        vx_vec = []
        vy_vec = []
        vz_vec = []

        for droneObject in self.droneObjects:
            droneNames.append(droneObject.droneName)
            # print("action = ", droneObject.currentAction)
            vx_val = droneObject.currentAction[0].item()
            vy_val = droneObject.currentAction[1].item()
            vz_val = droneObject.currentAction[2].item()

            # vx_vec.append(vx_val if not droneObject.reseting else 0)
            # vy_vec.append(vy_val if not droneObject.reseting else 0)
            # vz_vec.append(vz_val if not droneObject.reseting else 0)
            vx_vec.append(1 if not droneObject.reseting else 0)
            vy_vec.append(1 if not droneObject.reseting else 0)
            vz_vec.append(1 if not droneObject.reseting else 0)

            droneObject.currentStep += 1

        Utils.getClient().simPause(False)

        for i in range(len(droneNames)):
            # if i == 1:
            #     print(vx_vec[i], vy_vec[i], vz_vec[i])
            #     print(self.droneObjects[i].currentStatePos)
            Utils.getClient().moveByVelocityBodyFrameAsync(
                vx_vec[i],
                vy_vec[i],
                vz_vec[i],
                self.actionTime,
                vehicle_name=droneNames[i],
            )

        Utils.getClient().simPause(True)

    def resetBatch(self):
        windows = False

        # Size difference: -7710.0, -6070.0
        Utils.getClient().simPause(False)
        Utils.getClient().reset()
        time.sleep(5) if windows else time.sleep(0.25)

        start_poses = []
        # goal_poses = []
        for i in range(len(self.droneObjects)):
            start_pos, goal_pos = self.sample_start_goal_pos()

            self.droneObjects[i].currentGoal = goal_pos
            start_poses.append(start_pos)

        # airsim.Quaternionr(0.0, 0.0, 1.0, 0.0) = 180 degrees
        poses = [
            airsim.Pose(
                airsim.Vector3r(*start_poses[i]),
                airsim.Quaternionr(0.0, 0.0, 0.0, 0.0),
            )
            for i in range(len(self.droneObjects))
        ]

        for p, droneObject in zip(poses, self.droneObjects):
            print(p, droneObject.droneName)
            Utils.getClient().simSetVehiclePose(
                p, ignore_collision=True, vehicle_name=droneObject.droneName
            )

        time.sleep(5) if windows else time.sleep(0.25)

        for droneObject in self.droneObjects:
            Utils.getClient().armDisarm(True, droneObject.droneName)
            Utils.getClient().enableApiControl(True, droneObject.droneName)
            Utils.getClient().takeoffAsync(vehicle_name=droneObject.droneName)
            if windows:
                time.sleep(1)

            # Move up 3m
        time.sleep(5) if windows else time.sleep(0.25)

        for droneObject in self.droneObjects:
            quad_position = (
                Utils.getClient()
                .getMultirotorState(droneObject.droneName)
                .kinematics_estimated.position
            )
            # print(quad_position, start_pos)
            # Utils.getClient().takeoffAsync(vehicle_name=droneObject.droneName).join()
            # Utils.getClient().hoverAsync(vehicle_name=droneObject.droneName).join()
            Utils.getClient().moveToPositionAsync(
                quad_position.x_val,
                quad_position.y_val,
                self.constant_z_pos,
                3.0,
                vehicle_name=droneObject.droneName,
            )
            droneObject.currentStep = 0
            currentPos = (
                Utils.getClient()
                .getMultirotorState(droneObject.droneName)
                .kinematics_estimated.position.to_numpy_array()
            )
            droneObject.distanceFromGoal = np.linalg.norm(
                droneObject.currentGoal - currentPos
            )

            droneObject.reseting = False
            droneObject.currentTotalReward = 0
            if windows:
                time.sleep(1)

        # time.sleep(5)
        self.gatherAllObservations()
        time.sleep(5) if windows else time.sleep(0.25)

        Utils.getClient().simPause(True)
        self.episodes += 1

    def is_collision(self, droneObject: DroneObj):
        collisionInfo = Utils.getClient().simGetCollisionInfo(droneObject.droneName)

        hasCollided = collisionInfo.has_collided

        return hasCollided

    def is_landing(self, droneObject: DroneObj):
        # Set a threshold for how close the drone should be to the ground
        # to consider it landed

        landing_threshold = -0.1  # You may need to adjust this value
        state = Utils.getClient().getMultirotorState(droneObject.droneName)
        position = state.kinematics_estimated.position
        return position.z_val > landing_threshold

    def calculateReward(self, droneObject: DroneObj):
        image = droneObject.currentState["image"]
        reward = 0
        done = False

        drone_pos = (
            Utils.getClient()
            .getMultirotorState(droneObject.droneName)
            .kinematics_estimated.position.to_numpy_array()
        )

        potential_reward_weight = 0.20  # TODO: add in config file
        target_dist_curr = float(np.linalg.norm(droneObject.currentGoal - drone_pos))
        reward += (
            droneObject.distanceFromGoal - target_dist_curr
        ) * potential_reward_weight
        droneObject.distanceFromGoal = target_dist_curr

        info = {}

        if target_dist_curr < self.goal_threshold:
            reward += 1
            done = True
            info["is_success"] = True

        if self.is_collision(droneObject):
            print("The drone has collided with the obstacle!!!")
            reward += -1
            info["is_collision"] = True
            done = True
        elif self.is_landing(droneObject):
            # Check if the drone's altitude is less than the landing threshold
            print("Drone has touched the ground!!!")
            reward += -1
            done = True
        elif target_dist_curr >= 50:
            print("The drone has flown out of the specified range!!!")
            reward += -1
            done = True
        elif droneObject.currentStep > 100:
            info["is_timeout"] = True
            reward += -1
            done = True

        droneObject.currentTotalReward += reward
        return reward, done

    def resetStep(self, droneObject: DroneObj):
        if droneObject.reseting:
            if (
                droneObject.resetTick == 0
                and time.perf_counter() - droneObject.resetingTime > 1
            ):
                # This first step moves the vehicle?
                print(
                    "RESETING DRONE ",
                    droneObject.droneId,
                    print("len "),
                    len(self.droneObjects),
                )
                start_pos, goal_pos = self.sample_start_goal_pos()

                droneObject.currentGoal = goal_pos
                droneObject.start_pos = start_pos

                self.agent_start_pos = np.array(start_pos, dtype=np.float64)
                # print("self.agent_start_pos: ", self.agent_start_pos)

                p = airsim.Pose(
                    airsim.Vector3r(*self.agent_start_pos),
                    airsim.Quaternionr(0.0, 0.0, 0.0, 1.0),
                )
                Utils.getClient().simSetVehiclePose(
                    p, ignore_collision=True, vehicle_name=droneObject.droneName
                )

                # Utils.getClient().client.call_async(
                #     "resetVehicle",
                #     droneObject.droneName,
                #
                # )
                # Utils.getClient

                droneObject.resetTick = 1
                droneObject.resetingTime = time.perf_counter()

            if (
                droneObject.resetTick == 1
                and time.perf_counter() - droneObject.resetingTime > 1
            ):
                # Arm and enable
                # Utils.getClient().takeoffAsync(vehicle_name=droneObject.droneName)

                droneObject.resetingTime = droneObject.resetingTime
                droneObject.resetTick = 3

            if (
                droneObject.resetTick
                == 3
                # and time.perf_counter() - droneObject.resetingTime > 2
            ):
                # Move to estimated state?
                state = Utils.getClient().getMultirotorState(droneObject.droneName)

                dist = np.linalg.norm(
                    droneObject.start_pos
                    - state.kinematics_estimated.position.to_numpy_array()
                )
                if dist < 1:
                    Utils.getClient().armDisarm(True, droneObject.droneName)
                    Utils.getClient().enableApiControl(True, droneObject.droneName)
                    droneObject.reseting = False
                    droneObject.resetTick = 0

                if time.perf_counter() - droneObject.resetingTime > 5:
                    droneObject.resetTick = 0

                quad_position = state.kinematics_estimated.position
                # import pdb; pdb.set_trace()
                print("dist: ", dist)
                print("droneObject.start_pos: ", droneObject.start_pos)
                print("quad position: ", quad_position)
                # Utils.getClient().moveToPositionAsync(
                #     quad_position.x_val,
                #     quad_position.y_val,
                #     self.constant_z_pos,
                #     3.0,
                #     vehicle_name=droneObject.droneName,
                # )
                # Initialise the distance from goal
                current_pos = state.kinematics_estimated.position.to_numpy_array()
                droneObject.distanceFromGoal = np.linalg.norm(
                    droneObject.currentGoal - current_pos
                )

                droneObject.currentStep = 0
                droneObject.currentTotalReward = 0
                self.episodes += 1

    def tick(self, agent):
        for droneObject in self.droneObjects:
            if (
                float(
                    np.linalg.norm(
                        droneObject.currentGoal - droneObject.currentStatePos
                    )
                )
                < self.goal_threshold
            ):
                droneObject.reseting = True

            self.resetStep(droneObject)

            if not droneObject.reseting:
                action = agent.choose_action(droneObject.currentState)
                droneObject.currentAction = action.cpu().data.numpy()[0]

        self.doActionBatch()
        self.gatherAllObservations()

        loadDQNFile = False

        for droneObject in self.droneObjects:
            if not droneObject.reseting:
                self.numImagesSent += 1
                reward, done = self.calculateReward(droneObject)
                Utils.getModelServer().call_async(
                    "pushMemory",
                    Utils.convertStateDicToListDic(droneObject.previousState),
                    droneObject.currentAction.tolist(),  # was considered np.int rather than int.
                    Utils.convertStateDicToListDic(droneObject.currentState),
                    reward,
                    1 - int(done),
                )

                if done:
                    Utils.getModelServer().call_async(
                        "finishEpisode",
                        droneObject.distanceFromGoal,
                        droneObject.currentTotalReward,
                    )
                    droneObject.reseting = True
                    droneObject.resetingTime = time.perf_counter()
                    agent.epsilon = Utils.getModelServer().call("getEpsilon")
                    agent.memory.pushCounter = Utils.getModelServer().call(
                        "getMemoryPushCounter"
                    )
                    loadDQNFile = True

        if loadDQNFile and exists(
            "{}/ModelSaves/dqn.pth".format(pathlib.Path().resolve())
        ):
            try:
                os.rename(
                    "{}/ModelSaves/dqn.pth".format(pathlib.Path().resolve()),
                    "{}/ModelSaves/dqn_read.pth".format(pathlib.Path().resolve()),
                )
                agent.load(
                    "{}/ModelSaves/dqn_read.pth".format(pathlib.Path().resolve())
                )
                os.rename(
                    "{}/ModelSaves/dqn_read.pth".format(pathlib.Path().resolve()),
                    "{}/ModelSaves/dqn.pth".format(pathlib.Path().resolve()),
                )
            except:
                print("issue reading file")

        # print("NumImagesSent: ", self.numImagesSent)

        finished = True
        for droneObject in self.droneObjects:
            if not droneObject.reseting:
                finished = False

        finished = False

        return finished


# libUE4Editor-AirSim.so!_ZNSt3__110__function6__funcIZN3rpc6detail10dispatcher4bindIZN3msr6airlib22MultirotorRpcLibServerC1EPNS7_11ApiProviderENS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEEtE4$_14EEvRKSG_T_RKNS3_4tags14nonvoid_resultERKNSL_11nonzero_argEEUlRKN14clmdep_msgpack2v26objectEE_NSE_ISX_EEFNS_10unique_ptrINSS_2v113object_handleENS_14default_deleteIS11_EEEESW_EEclESW_()

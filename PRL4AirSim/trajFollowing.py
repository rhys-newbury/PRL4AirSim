import argparse
import os
import airsim
import subprocess
import psutil
import time
import numpy as np
import math
import matplotlib.pyplot as plt

import sys

# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from API import airsimUtils
from TrajFollow import TrajectoryFollower, Spline

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)


def main(args):

    ######### UNREAL ENGINE / AIRSIM SETUP #########
    if args.useAirsim:
        print("Starting Unreal Engine")
        try:
            unrealEngineProcess = subprocess.Popen(
                [args.unrealPath, "-WINDOWED -ResX=640 -ResY=480"],
                stdout=subprocess.DEVNULL,
            )
            print("Spawned UE4 process with PID: ", unrealEngineProcess.pid)
        except Exception as e:
            print(
                f"Failed to spawn UE4, make sure you have correctly set the --unrealPath argument to point to the correct binary"
            )
            print(e)
            exit()
        time.sleep(2.5)

        print("Connecting to AirSim")
        client = airsim.MultirotorClient(port=args.UEport)
        print("yeet??")
        # client.confirmConnection()
        client.enableApiControl(True, vehicle_name="Drone0")
        success = client.armDisarm(True, vehicle_name="Drone0")
        if not success:
            print("Failed to arm drone")
            exit()

        print("Taking off")
        client.takeoffAsync(vehicle_name="Drone0").join()

    ######### DEFINE TRAJECTORY #########
    trajectoryPoints = []
    if args.useAirsim:
        initialPosition = client.simGetVehiclePose(vehicle_name="Drone0").position
        initialPosition = np.array(
            [initialPosition.x_val, initialPosition.y_val, initialPosition.z_val]
        )
    else:
        initialPosition = np.array([0.0, 0.0, 0.0])
    trajectoryPoints.append(initialPosition)

    if args.trajectoryType == "square":
        trajectoryPoints.append(initialPosition + np.array([5.0, 0.0, -2.0]))
        trajectoryPoints.append(initialPosition + np.array([5.0, 5.0, -2.0]))
        trajectoryPoints.append(initialPosition + np.array([0.0, 5.0, -2.0]))
        trajectoryPoints.append(initialPosition + np.array([0.0, 0.0, -2.0]))

        trajectoryPoints.append(initialPosition + np.array([5.0, 0.0, -4.0]))
        trajectoryPoints.append(initialPosition + np.array([5.0, 5.0, -4.0]))
        trajectoryPoints.append(initialPosition + np.array([0.0, 5.0, -4.0]))
        trajectoryPoints.append(initialPosition + np.array([0.0, 0.0, -4.0]))

    elif args.trajectoryType == "zigzag":
        trajectoryPoints.append(initialPosition + np.array([-5.0, 5.0, -2.0]))
        trajectoryPoints.append(initialPosition + np.array([-10.0, 0.0, -3.0]))
        trajectoryPoints.append(initialPosition + np.array([-15.0, 5.0, -2.0]))
        trajectoryPoints.append(initialPosition + np.array([-20.0, 0.0, -3.0]))
        trajectoryPoints.append(initialPosition + np.array([-25.0, 5.0, -2.0]))
        trajectoryPoints.append(initialPosition + np.array([-30.0, 0.0, -3.0]))

    elif args.trajectoryType == "circle":
        for i in range(0, 360 * 2, 30):
            trajectoryPoints.append(
                initialPosition
                + np.array(
                    [
                        5.0 * np.cos(np.deg2rad(i)),
                        5.0 * np.sin(np.deg2rad(i)),
                        -2.0 + i / 360,
                    ]
                )
            )

    ######### GENERATE TRAJECTORY #########
    trajectoryFollower = TrajectoryFollower(trajectoryPoints, 2.0)
    trajectoryFollower.generateTrajectory(
        splinePower=args.splinePower,
        intermediatePoints=args.intermediatePoints,
        additionalPower=args.additionalPower,
    )

    ######### EXECUTE TRAJECTORY #########
    if args.useAirsim:
        recordedPositions = []
        recordedVelocities = []

        actionTime = 1
        flying = True
        # client.simPause(False)

        # client.reset()
        time.sleep(0.25)
        p = airsim.Pose(
            airsim.Vector3r(-14, 16, -3),
            airsim.Quaternionr(1.0, 0.0, 0.0, 0.0),
        )
        for _ in range(3):
            
            
            client.simSetVehiclePose(
                p, ignore_collision=True, vehicle_name="Drone0"
            )
            # client.client.call("simSetVehiclePoseBatch", [p], ["Drone0"])
            time.sleep(2)

            client.enableApiControl(True, vehicle_name="Drone0")
            success = client.armDisarm(True, vehicle_name="Drone0")
            # Utils.getClient().takeoffAsync(vehicle_name=droneObject.droneName)


            # client.moveToPositionAsync(p.position.x_val, p.position.y_val, p.position.z_val, velocity=5.0).join()

            if not success:
                print("Failed to arm drone")
                exit()

            # client.simPause(False)

            print("Taking off")
            client.takeoffAsync(vehicle_name="Drone0").join()
            
            # client.moveToPositionAsync(-14, 16, -3, 3.0, vehicle_name="Drone0").join()

            state = client.getMultirotorState(vehicle_name="Drone0")
            currentPosition = state.kinematics_estimated.position
            print(currentPosition)

            # client.simPause(True)
            

            time.sleep(4)

            for x in range(20):
                print(x)
            # while flying:

                # Get the state of the drone, trajectory follower requires current position & velocity (and yaw for yaw control)
                # quaternion = np.array(
                #     [
                #         state.kinematics_estimated.orientation.w_val,
                #         state.kinematics_estimated.orientation.x_val,
                #         state.kinematics_estimated.orientation.y_val,
                #         state.kinematics_estimated.orientation.z_val,
                #     ],
                #     dtype=np.float32,
                # )
                # currentYaw = math.atan2(
                #     2 * (quaternion[0] * quaternion[3] + quaternion[1] * quaternion[2]),
                #     1 - 2 * (quaternion[2] ** 2 + quaternion[3] ** 2),
                # )

                # # Get the next action from the trajectory follower
                # velocity, yawRate, flying = trajectoryFollower.getAction(
                #     currentPosition, currentVelocity, currentYaw=currentYaw
                # )
                # print(currentPosition)
                # client.client.call("simPause", False)

                #  yaw_mode=airsim.YawMode(True, yawRate * 180 / math.pi),
                # Execute the action
                # client.moveByVelocityAsync(velocity[0], velocity[1], velocity[2], actionTime * 2, vehicle_name='Drone0')
                client.moveByVelocityAsync(2, 3, -5, actionTime * 2, vehicle_name='Drone0')
                state = client.getMultirotorState(vehicle_name="Drone0")
                currentPosition = state.kinematics_estimated.position
                currentVelocity = state.kinematics_estimated.linear_velocity
                currentPosition = np.array(
                    [currentPosition.x_val, currentPosition.y_val, currentPosition.z_val]
                )
                currentVelocity = np.array(
                    [currentVelocity.x_val, currentVelocity.y_val, currentVelocity.z_val]
                )
                print(currentPosition)
                print(currentVelocity)

                # client.client.call_async(
                #     "moveByVelocityZBatch",
                #     [velocity[0]],
                #     [velocity[1]],
                #     [velocity[2]],
                #     actionTime * 2,
                #     airsim.DrivetrainType.MaxDegreeOfFreedom,
                #     airsim.YawMode(),
                #     ["Drone0"],
                # ).join()

                # Log and sleep till next action
                recordedPositions.append(currentPosition)
                # recordedVelocities.append(currentVelocity)
                time.sleep(actionTime)

                # client.client.call("simPause", True)

            client.client.call("simPause", False)

        # client.enableApiControl(False, vehicle_name='Drone0')
        # success = client.armDisarm(False, vehicle_name='Drone0')

        # client.enableApiControl(True, vehicle_name='Drone0')
        # success = client.armDisarm(True, vehicle_name='Drone0')

        # client.takeoffAsync(vehicle_name='Drone0').join()

        # client.moveToPositionAsync(p.position.x_val, p.position.y_val, p.position.z_val, velocity=5.0).join()
        # while True:
        #     currentPosition = state.kinematics_estimated.position
        #     print(currentPosition, p.position)
        # .join()

        time.sleep(1.0)
        print("Killing Unreal Engine")
        killUE(unrealEngineProcess)

    ######### PLOT RESULTS ####qz#####
    # targetPositions = [x.positions for x in trajectoryFollower.trajectory]
    # targetVelocities = [x.velocities for x in trajectoryFollower.trajectory]

    positionPlot = plt.figure().add_subplot(projection="3d")
    # for spline in targetPositions:
    #     positionPlot.plot(
    #         spline[:, 0], spline[:, 1], spline[:, 2], label="Target positions"
    #     )

    if args.useAirsim:
        recordedPositions = np.array(recordedPositions)
        recordedVelocities = np.array(recordedVelocities)
    # trajectoryPoints = np.array(trajectoryPoints)

    if args.useAirsim:
        positionPlot.scatter(
            recordedPositions[:, 0],
            recordedPositions[:, 1],
            recordedPositions[:, 2],
            label="Recorded positions",
        )
    # positionPlot.scatter(
    #     trajectoryPoints[:, 0],
    #     trajectoryPoints[:, 1],
    #     trajectoryPoints[:, 2],
    #     c="r",
    #     marker="x",
    #     label="Trajectory waypoints",
    # )
    # positionPlot.set_title("Position")

    # velocityPlot = plt.figure().add_subplot(projection="3d")
    # for spline in targetVelocities:
    #     velocityPlot.plot(
    #         spline[:, 0], spline[:, 1], spline[:, 2], label="Target velocities"
    #     )

    # if args.useAirsim:
    #     velocityPlot.scatter(
    #         recordedVelocities[:, 0],
    #         recordedVelocities[:, 1],
    #         recordedVelocities[:, 2],
    #         label="Recorded velocities",
    #     )
    # velocityPlot.set_title("Velocity")

    plt.show()
    print("Finished")


def killUE(unrealEngineProcess):

    try:
        UEParent = psutil.Process(unrealEngineProcess.pid)
        UEChildren = UEParent.children(recursive=True)
        for child in UEChildren:
            child.kill()

        gone, still_alive = psutil.wait_procs(UEChildren, timeout=5)
        if len(still_alive) > 0:
            print("FAILED TO KILL CHILD: ", still_alive, ", was able to kill: ", gone)

        if psutil.pid_exists(UEParent.pid):
            UEParent.kill()

    except Exception as e:
        print(f"Failed to kill UE4")
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to a demonstrate all the different sensors / perception capabilities of AirSim"
    )
    parser.add_argument(
        "--UEport",
        type=int,
        help="What port to use to connect to the unreal engine?",
        default=41451,
    )
    parser.add_argument(
        "--unrealPath", type=str, help="Where to load the unreal binary", default=""
    )
    parser.add_argument(
        "--useAirsim",
        type=bool,
        help="Whether to try the trajectory with the UAV or just generate a trajectory to plot",
        default=True,
    )

    # Path following arguments
    parser.add_argument(
        "--splinePower",
        type=int,
        help="What power of a spline to use to fit the trajectory waypoint (1=linear, 2=quadratic, 3.., -1 for single contunous spline)",
        default=3,
    )
    parser.add_argument(
        "--intermediatePoints",
        type=int,
        help="How many intermediate points to use between each trajectory waypoint",
        default=6,
    )
    parser.add_argument(
        "--additionalPower",
        type=int,
        help="How many additional powers to add to the spline to account for the additional intermediate points",
        default=4,
    )

    # Trajectory selection
    parser.add_argument(
        "--trajectoryType",
        type=str,
        help="What trajectory to follow (square, zigzag, circle)",
        default="square",
    )

    args = parser.parse_args()

    if args.intermediatePoints < args.additionalPower:
        print(
            "The additional power must not be greater than the additional intermediate points"
        )
        exit()

    main(args)

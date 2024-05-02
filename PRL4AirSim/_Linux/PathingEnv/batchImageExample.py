import airsim
import cv2
import argparse
import os
import json
import time
import numpy as np

def main(args):
    startHeight = -20
    droneSpacing = -4

    createSettings(args.settingSaveDir, args.nDrones)

    client = airsim.MultirotorClient(ip="127.0.0.1", port=29001)
    client.confirmConnection()

    time.sleep(1.0)
    isDroneDescending = []
    for drone in range(args.nDrones):
        client.enableApiControl(True, vehicle_name=f"Drone{drone}")
        client.armDisarm(True, vehicle_name=f"Drone{drone}")
        client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0.0, 0.0, startHeight + droneSpacing * drone), airsim.to_quaternion(0, 0, 0)), True, vehicle_name=f"Drone{drone}")
        client.takeoffAsync(vehicle_name=f"Drone{drone}")
        isDroneDescending.append(True)

    time.sleep(2.0)

    imageMessage = []
    vehicleNames = []
    for i in range(args.nDrones):
        vehicleNames.append(f"Drone{i}")
        imageMessage.append(airsim.ImageRequest("DownwardsCamera", airsim.ImageType.Scene, pixels_as_float=False, compress=False))

    print("Taking batch images now...")
    while True:
        images = client.client.call("ANSR_simGetBatchImages", imageMessage, vehicleNames)

        for imageID, image in enumerate(images):
            numpyImage = np.frombuffer(image['image_data_uint8'], dtype=np.uint8).reshape(image['height'], image['width'], 3)
            cv2.imshow(f"Drone{imageID}", numpyImage)

        for drone in range(args.nDrones):
            pose = client.simGetVehiclePose(vehicle_name=f"Drone{drone}")
            if pose.position.z_val > startHeight:
                isDroneDescending[drone] = False
            if pose.position.z_val < startHeight + droneSpacing * (args.nDrones - 1):
                isDroneDescending[drone] = True

            if isDroneDescending[drone]:
                client.moveByVelocityAsync(0, 0, 1, 1, vehicle_name=f"Drone{drone}")
            else:
                client.moveByVelocityAsync(0, 0, -1, 1, vehicle_name=f"Drone{drone}")

        cv2.waitKey(1)

def createSettings(saveDir, nDrones):

    settingsDict = {
            "SettingsVersion": 1.2,
            "SimMode": "Multirotor",
        }

    cameraDict = {
        "DownwardsCamera": {
            "CaptureSettings": [
                {
                    "FOV_Degrees": 90,
                    "ImageType": 0,
                    "Width": 640,
                    "Height": 480
                }
            ],
            "Pitch": -90.0,
            "Yaw": 0.0,
            "Roll": 0.0,
            "X": 0.0,
            "Y": 0.0,
            "Z": 0.2
        }
    }

    vehiclesDict = {}
    for i in range(nDrones):
        vehiclesDict[f"Drone{i}"] = {"VehicleType": "SimpleFlight", "Cameras": cameraDict}
    settingsDict["Vehicles"] = vehiclesDict

    with open(os.path.join(saveDir, 'settings.json'), 'w') as fp:
        json_string = json.dumps(settingsDict, default=lambda o: o.__dict__, sort_keys=True, indent=2)
        fp.write(json_string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to define pathing environments")

    # Read / write parameters
    parser.add_argument('--settingSaveDir', type=str, help='Directory to save settings.json file', default="")
    parser.add_argument('--nDrones', type=int, help='Number of drones to spawn', default=6)
    args = parser.parse_args()

    if args.settingSaveDir == "":
        print("Please specify a directory to save the settings.json file to")
        exit()

    main(args)




import os
import json
import os
import pathlib
import time
import Utils

homeDir = str(pathlib.Path.home())
projectName = Utils.getConfig()["projectName"]
envProcesses = int(Utils.getConfig()["envProcesses"])
storage_port = int(Utils.getConfig()["storage_port"])
headless = bool(Utils.getConfig()["headless"])


def run_command(cmd):
    print(cmd)
    os.system(cmd + "; read -p 'Press Enter to exit'\"")


def changeUEIPJson(port):
    with open(
        str(pathlib.Path.home() / "Documents" / "AirSim" / "settings.json"), "r"
    ) as jsonFile:
        data = json.load(jsonFile)

    data["ApiServerPort"] = port

    with open(
        str(pathlib.Path.home() / "Documents" / "AirSim" / "settings.json"), "w"
    ) as jsonFile:
        json.dump(data, jsonFile, indent=4)


run_command('gnome-terminal -- bash -c "python Storage.py {}'.format(storage_port))
time.sleep(5)
run_command('gnome-terminal -- bash -c "python Trainer.py {}'.format(storage_port))

for i in range(envProcesses):
    port = storage_port + i + 1
    changeUEIPJson(port)
    if headless:
        run_command(
            'gnome-terminal -- bash -c "./Linux/{projectName}.sh -RenderOffscreen -windowed -NoVSync'.format(
                projectName=projectName
            )
        )
    else:
        windowX = 1000 * i
        windowY = 1000

        run_command(
            'gnome-terminal -- bash -c "./Linux/{projectName}.sh -windowed -WinX={WinX} -WinY={WinY} -NoVSync'.format(
                projectName=projectName, WinX=windowX, WinY=windowY
            )
        )
    time.sleep(4)

time.sleep(5)
for i in range(envProcesses):
    UE_port = storage_port + i + 1
    run_command(
        'gnome-terminal -- bash -c "python PyClient.py {UE_port} {UE_Address} {storage_port}'.format(
            UE_port=UE_port, UE_Address="127.0.0.1", storage_port=storage_port
        )
    )

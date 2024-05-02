import argparse
import json
import os

def generateEnvironment(args):

    if args.environmentBrightness < 0.0 or args.environmentBrightness > 1.0:
        raise ValueError('Brightness must be between 0.0 and 1.0')

    if args.environmentSunYaw < 0.0 or args.environmentSunYaw > 360.0:
        raise ValueError('Sun yaw must be between 0.0 and 360.0')

    if args.environmentSunPitch < 0.0 or args.environmentSunPitch > 90.0:
        raise ValueError('Sun pitch must be between 0.0 and 90.0')

    if args.environmentType == 'forest':

        if args.forestDensity < 0.0 or args.forestDensity > 1.0:
            raise ValueError('Forest density must be between 0.0 and 1.0')

        if args.forestMinHeight > args.forestMaxHeight:
            raise ValueError('Forest minimum height must be less than or equal to the maximum height')

        if args.forestMinHeight < 0.0:
            raise ValueError('Forest minimum height must be greater than 0.0')

        if args.forestMaxHeight < 0.0:
            raise ValueError('Forest maximum height must be greater than 0.0')

        if args.forestEnableBranches:
            if args.forestBranchesMinLength > args.forestBranchesMaxLength:
                raise ValueError('Forest branches minimum length must be less than or equal to the maximum length')

            if args.forestBranchesMinLength < 0.0:
                raise ValueError('Forest branches minimum length must be greater than 0.0')

            if args.forestBranchesMaxLength < 0.0:
                raise ValueError('Forest branches maximum length must be greater than 0.0')

            if args.forestBranchesDensity < 0.0 or args.forestBranchesDensity > 1.0:
                raise ValueError('Forest branches density must be between 0.0 and 1.0')

    elif args.environmentType == 'grid':

        if args.gridSize < 0.0:
            raise ValueError('Grid size must be greater than 0.0')

        if args.gridSpacingSize < 0.0:
            raise ValueError('Grid spacing size must be greater than 0.0')

        if args.gridMinHeight > args.gridMaxHeight:
            raise ValueError('Grid minimum height must be less than or equal to the maximum height')

        if args.gridMinHeight < 0.0:
            raise ValueError('Grid minimum height must be greater than 0.0')

        if args.gridMaxHeight < 0.0:
            raise ValueError('Grid maximum height must be greater than 0.0')

        if args.gridEnableTrees:
            if args.gridTreeDensity < 0.0 or args.gridTreeDensity > 1.0:
                raise ValueError('Grid tree density must be between 0.0 and 1.0')

        if args.gridEnableBridges:
            if args.gridBridgeDensity < 0.0 or args.gridBridgeDensity > 1.0:
                raise ValueError('Grid bridge density must be between 0.0 and 1.0')

    elif args.environmentType == 'maze':

        if args.mazeSizeX < 3:
            raise ValueError('Maze size X must be greater than 3')

        if args.mazeSizeY < 3:
            raise ValueError('Maze size Y must be greater than 3')

    if args.environmentDynamic:
        if args.environmentDynamicDensity < 0.0 or args.environmentDynamicDensity > 1.0:
            raise ValueError('Dynamic density must be between 0.0 and 1.0')


    # Create dictionary to store environment information
    environment = {}
    environment['environmentType'] = args.environmentType
    environment['environmentWidth'] = args.environmentWidth
    environment['environmentLength'] = args.environmentLength
    environment['environmentDynamic'] = args.environmentDynamic
    environment['environmentBrightness'] = args.environmentBrightness
    environment['environmentSunYaw'] = args.environmentSunYaw
    environment['environmentSunPitch'] = args.environmentSunPitch
    if args.environmentDynamic:
        environment['environmentDynamicDensity'] = args.environmentDynamicDensity

    if args.environmentType == 'forest':
        environment['forestDensity'] = args.forestDensity
        environment['forestMinHeight'] = args.forestMinHeight
        environment['forestMaxHeight'] = args.forestMaxHeight

        environment['forestEnableBranches'] = args.forestEnableBranches
        if args.forestEnableBranches:
            environment['forestBranchesMinLength'] = args.forestBranchesMinLength
            environment['forestBranchesMaxLength'] = args.forestBranchesMaxLength
            environment['forestBranchesDensity'] = args.forestBranchesDensity

    elif args.environmentType == 'grid':
        environment['gridSize'] = args.gridSize
        environment['gridSpacingSize'] = args.gridSpacingSize
        environment['gridMinHeight'] = args.gridMinHeight
        environment['gridMaxHeight'] = args.gridMaxHeight

        environment['gridEnableTrees'] = args.gridEnableTrees
        if args.gridEnableTrees:
            environment['gridTreeDensity'] = args.gridTreeDensity

        environment['gridEnableBridges'] = args.gridEnableBridges
        if args.gridEnableBridges:
            environment['gridBridgeDensity'] = args.gridBridgeDensity

    elif args.environmentType == 'maze':
        environment['mazeSizeX'] = args.mazeSizeX
        environment['mazeSizeY'] = args.mazeSizeY


    # Save environment information to file
    with open(os.path.join(args.saveDir, 'environment.json'), 'w') as f:
        json_string = json.dumps(environment, default=lambda o: o.__dict__, sort_keys=True, indent=2)
        f.write(json_string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to define pathing environments")

    # Read / write parameters
    parser.add_argument('--saveDir', type=str, help='Directory to save environment information to', default="")

    # Environment parameters
    parser.add_argument('--environmentType', type=str, help='What type of environment to generate', choices=['forest', 'grid', 'maze'], default='grid')
    parser.add_argument('--environmentWidth', type=float, help='Width of the environment (m)', default=100.0)
    parser.add_argument('--environmentLength', type=float, help='Length of the environment (m)', default=100.0)
    parser.add_argument('--environmentDynamic', type=bool, help='Whether the environment contains dynamic objects (True) or static (False)', default=True)
    parser.add_argument('--environmentDynamicDensity', type=float, help='Density of dynamic objects (range of 0.0 - 1.0)', default=0.5)

    parser.add_argument('--environmentBrightness', type=float, help='Brightness of the environment (range of 0.0 - 1.0)', default=1.0)
    parser.add_argument('--environmentSunYaw', type=float, help='Yaw of the sun (degrees)', default=50.0)
    parser.add_argument('--environmentSunPitch', type=float, help='Pitch of the sun (degrees)', default=45.0)

    # Forest parameters
    parser.add_argument('--forestDensity', type=float, help='Density of obstacles (range of 0.0 - 1.0)', default=0.5)
    parser.add_argument('--forestMinHeight', type=float, help='Minimum height of obstacles (m)', default=1.0)
    parser.add_argument('--forestMaxHeight', type=float, help='Maximum height of obstacles (m)', default=10.0)

    parser.add_argument('--forestEnableBranches', type=bool, help='Whether to enable horizontal branches (True) or not (False)', default=True)
    parser.add_argument('--forestBranchesMinLength', type=float, help='Minimum length of horizontal branches (m)', default=5.0)
    parser.add_argument('--forestBranchesMaxLength', type=float, help='Maximum length of horizontal branches (m)', default=20.0)
    parser.add_argument('--forestBranchesDensity', type=float, help='Density of branches (range of 0.0 - 1.0)', default=0.5)

    # Grid parameters
    parser.add_argument('--gridSize', type=float, help='Size of grid cells (m)', default=15.0)
    parser.add_argument('--gridSpacingSize', type=float, help='Spacing between grid cells (m)', default=8.0)
    parser.add_argument('--gridMinHeight', type=float, help='Minimum height of buildings (m)', default=5.0)
    parser.add_argument('--gridMaxHeight', type=float, help='Maximum height of buildings (m)', default=20.0)

    parser.add_argument('--gridEnableTrees', type=bool, help='Whether to enable trees (True) or not (False)', default=True)
    parser.add_argument('--gridTreeDensity', type=float, help='Density of trees (range of 0.0 - 1.0)', default=0.5)

    parser.add_argument('--gridEnableBridges', type=bool, help='Whether to enable bridges (True) or not (False)', default=True)
    parser.add_argument('--gridBridgeDensity', type=float, help='Density of bridges (range of 0.0 - 1.0)', default=0.5)

    # Maze parameters
    parser.add_argument('--mazeSizeX', type=int, help='How many cells should the maze be in width', default=21)
    parser.add_argument('--mazeSizeY', type=int, help='How many cells should the maze be in length', default=21)

    args = parser.parse_args()
    generateEnvironment(args)
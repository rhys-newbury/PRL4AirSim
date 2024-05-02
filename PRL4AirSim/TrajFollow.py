import numpy as np
import math

class Spline():
    def __init__(self, coefficients, timeLength, dt=0.01):
        self.coefficients = coefficients
        self.timeLength = timeLength
        self.dt = dt

        self.positions = np.zeros((int(self.timeLength / self.dt), 3))
        self.velocities = np.zeros((int(self.timeLength / self.dt), 3))

        # Calculate positions along spline at each time step
        power = np.linspace(0, len(self.coefficients[0]) - 1, len(self.coefficients[0])).reshape(-1, 1)
        for i in range(self.positions.shape[0]):
            self.positions[i, :] = (self.coefficients @ ((i * self.dt) ** power).reshape(-1))

        # Calculate velocities along spline at each time step via differentiation
        powerVel = power - 1
        for i in range(self.velocities.shape[0]):
            self.velocities[i, :] = (self.coefficients @ (((i * self.dt) ** powerVel) * power)).reshape(-1)
        self.velocities[0, :] = self.velocities[1, :]

    def getClosestPosition(self, position, region, timeRange=1.0):

        r0 = np.clip(region, 0, self.positions.shape[0] - 1)
        r1 = np.clip(region + int(timeRange/self.dt), 0, self.positions.shape[0] - 1)

        distances = np.linalg.norm(position - self.positions[r0:r1, :], axis=1)
        minIndex = np.argmin(distances) + r0
        return self.positions[minIndex, :], minIndex

class TrajectoryFollower:
    def __init__(self, trajectoryPoints, speed):
        self.trajectoryPoints = trajectoryPoints
        self.speed = speed

        self.trajectory = []
        self.trajectoryState = 0
        self.priorSplineRegion = 0

        self.lookAheadWeighting = 0.6
        self.pGain_CurrentPosition = 0.4
        self.pGain_CurrentVelocity = 0.1
        self.pGain_LookAheadPosition = 0.5
        self.pGain_LookAheadVelocity = 0.1
        self.pGain_Yaw = 0.5

    def generateTrajectory(self, splinePower=-1, intermediatePoints=0, additionalPower=0):

        if splinePower == -1: # Single continuous spline
            self.trajectory = [self.generateSpline(self.trajectoryPoints, intermediatePoints, additionalPower)]
            return True
        elif splinePower > 0: # Multiple sub-splines
            nSplines = (len(self.trajectoryPoints) - 1) // splinePower
            if (len(self.trajectoryPoints) - 1) % splinePower != 0:
                nSplines += 1

            for spline in range(nSplines):
                self.trajectory.append(self.generateSpline(self.trajectoryPoints[spline*splinePower:np.clip((spline+1)*splinePower+1, 0, len(self.trajectoryPoints))], intermediatePoints, additionalPower))
            return True
        else:
            print("Spline power must be greater than or equal to 1, OR -1 for a single continuous spline")
            return False

    def generateSpline(self, points, intermediatePoints=0, additionalPower=0):

        if additionalPower < intermediatePoints:
            power = len(points) - 1 + additionalPower
        else:
            print("The amount of intermediate points must be equal to or greater than the additional power required to raise the power of the spline")
            power = len(points) - 1 + intermediatePoints

        trajectoryPoints = np.zeros([len(points) + (len(points) - 1) * intermediatePoints, 3], dtype=np.float32)
        for i in range(len(points) - 1):
            trajectoryPoints[i * (intermediatePoints + 1)] = points[i]

            for j in range(intermediatePoints):
                trajectoryPoints[i * (intermediatePoints + 1) + j + 1, :] = points[i] + (points[i + 1] - points[i]) * (j + 1) / (intermediatePoints + 1)
        trajectoryPoints[-1, :] = points[-1]

        distances = [0]
        for i in range(len(trajectoryPoints) - 1):
            distances.append(np.linalg.norm(trajectoryPoints[i + 1] - trajectoryPoints[i]))

        times = np.cumsum(distances) / self.speed
        coefficients = np.zeros((3, power + 1))

        # Perform linear optimisation to find the coefficients of the spline for each axis
        for axis in range(3):
            x = np.zeros((trajectoryPoints.shape[0], power + 1))
            y = np.zeros((trajectoryPoints.shape[0], 1))

            for j in range(trajectoryPoints.shape[0]):

                x[j, 0] = 1
                for i in range(power):
                    x[j, i + 1] = times[j] ** (i+1)

                y[j, 0] = trajectoryPoints[j, axis]

            beta = np.linalg.pinv(x)
            coefficients[axis, :] = np.matmul(beta, y).reshape(-1)

        return Spline(coefficients, times[-1])

    def getAction(self, currentPosition, currentVelocity, currentYaw=None, lookAheadTime=1.0):

        # Get the closest position and velocity along the spline within a specific region
        currentPositionAlongSpline, splineIndex = self.trajectory[self.trajectoryState].getClosestPosition(currentPosition, self.priorSplineRegion)
        targetVelocityAlongSpline = self.trajectory[self.trajectoryState].velocities[splineIndex, :]

        currentPositionError = currentPositionAlongSpline - currentPosition
        currentVelocityError = targetVelocityAlongSpline - currentVelocity
        currentTargetVelocity = targetVelocityAlongSpline + currentPositionError * self.pGain_CurrentPosition + currentVelocityError * self.pGain_CurrentVelocity


        # Get look ahead position
        nextIndex = splineIndex + int(lookAheadTime / self.trajectory[self.trajectoryState].dt)
        if nextIndex < self.trajectory[self.trajectoryState].positions.shape[0]:
            lookAheadPosition = self.trajectory[self.trajectoryState].positions[nextIndex, :]
            lookAheadVelocity = self.trajectory[self.trajectoryState].velocities[nextIndex, :]
        else:
            self.trajectoryState += 1
            if self.trajectoryState >= len(self.trajectory):
                print("End of trajectory reached")
                return np.zeros(3), 0.0, False

            leftOverTime = lookAheadTime - (self.trajectory[self.trajectoryState].positions.shape[0] - splineIndex) * self.trajectory[self.trajectoryState].dt
            splineIndex = 0

            nextIndex = np.clip(int(leftOverTime / self.trajectory[self.trajectoryState].dt),0, self.trajectory[self.trajectoryState].positions.shape[0] - 1)
            lookAheadPosition = self.trajectory[self.trajectoryState].positions[nextIndex, :]
            lookAheadVelocity = self.trajectory[self.trajectoryState].velocities[nextIndex, :]

        lookAheadPositionError = lookAheadPosition - (currentPosition + currentVelocity * lookAheadTime)
        lookAheadVelocityError = lookAheadVelocity - currentVelocity
        lookAheadVelocity = lookAheadVelocity + lookAheadPositionError * self.pGain_LookAheadPosition + lookAheadVelocityError * self.pGain_LookAheadVelocity

        outputVelocity = currentTargetVelocity * (1 - self.lookAheadWeighting) + lookAheadVelocity * self.lookAheadWeighting


        # Calculate yaw rate
        if currentYaw is not None:
            targetYaw = math.atan2(outputVelocity[1], outputVelocity[0])
            yawError = targetYaw - currentYaw
            if yawError > math.pi:
                yawError -= 2 * math.pi
            elif yawError < -math.pi:
                yawError += 2 * math.pi
            outputYawRate = yawError * self.pGain_Yaw

        else:
            outputYawRate = 0.0

        self.priorSplineRegion = splineIndex
        return outputVelocity, outputYawRate, True


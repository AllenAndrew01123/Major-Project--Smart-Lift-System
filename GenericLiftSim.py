import time

class GenericLift:
    def __init__(self, floorLimit, personLimit, powerIdle, powerStandby, ratioIdle, ratioStandby, timeAverageTravel, numTrips, percentAvgDist, loadFactor, estRunningEnergy):
        self.floorLimit = floorLimit
        self.personLimit = personLimit
        self.currFloor = 0
        self.currPersonCount = 0
        self.queue = {}
        for floor in range(floorLimit + 1):
            self.queue[floor] = {
                'upRequests': [],
                'downRequests': []
            }
        self.state = 'idle'
        self.idleEnergyConsumption = []
        self.movingEnergyConsumption = []
        self.powerIdle = powerIdle
        self.powerStandby = powerStandby
        self.ratioIdle = ratioIdle
        self.ratioStandby = ratioStandby
        self.timeAvgTravel = timeAverageTravel
        self.numTrips = numTrips
        self.percentAvgDist = percentAvgDist
        self.loadFactor = loadFactor
        self.estRunningEnergy = estRunningEnergy


    def addRequest(self, presentFloor, direction, goToFloor, peopleCount, timeOfRequest):
        if direction == 'up':
            self.queue[presentFloor]['upRequests'].append((goToFloor, peopleCount, timeOfRequest))
        else:
            self.queue[presentFloor]['downRequests'].append((goToFloor, peopleCount, timeOfRequest)) 


    def recordIdleEnergy(self, idleAtFloor, timestamp):
        factor1 = 24 - (self.numTrips * self.timeAvgTravel) / 3600
        factor2 = self.powerIdle * self.ratioIdle
        factor3 = self.powerStandby * self.ratioStandby
        factor2_3 = factor2 + factor3
        factor1_2_3 = factor1 * factor2_3
        idleEnergy = factor1_2_3 + idleAtFloor * 50
        self.idleEnergyConsumption.append((idleEnergy, timestamp))


    def recordMovingEnergy(self, startFloor, endFloor, startTimestamp, endTimestamp):
        factor1 = (self.numTrips * self.percentAvgDist * self.loadFactor * self.estRunningEnergy) / 2
        movingEnergy = factor1 + (abs(startFloor - endFloor) + 1) * 75
        self.movingEnergyConsumption.append((movingEnergy, endTimestamp - startTimestamp))


    def requestsAvailableBool(self, currTimeStamp):
        requests = False
        if self.state == 'idle' or self.currPersonCount == self.personLimit:
            return requests
        if self.state == 'movingup':
            for floor in range(self.currFloor + 1, self.floorLimit + 1):
                if len(self.queue[floor]['upRequests']) > 0 or len(self.queue[floor]['downRequests']) > 0:
                    requests = True
                    break
        else:
            for floor in range(0, self.currFloor - 1):
                if len(self.queue[floor]['upRequests']) > 0 or len(self.queue[floor]['downRequests']) > 0:
                    requests = True
                    break
        if currTimeStamp < self.queue[0]['upRequests'][0]:
            requests = True
        return requests


    def absoluteRequestsBool(self):
        requests = False
        for floor in range(self.floorLimit + 1):
            if len(self.queue[floor]['upRequests']) > 0 or len(self.queue[floor]['downRequests']) > 0:
                    requests = True
                    break
        return requests


    def runSimulation(self, unitHaltDuration):
        ticks = 0
        while self.absoluteRequestsBool():
            if self.currPersonCount == 0:
                liveRequests = []
                for floor in range(0, self.floorLimit + 1):
                    if len(self.queue[floor]['upRequests']) != 0:
                        if self.queue[floor]['upRequests'][0][-1] <= ticks:
                            liveRequests.append((self.queue[floor]['upRequests'][0], floor, 'upRequests'))
                    if len(self.queue[floor]['downRequests']) != 0:
                        if self.queue[floor]['downRequests'][0][-1] <= ticks:
                            liveRequests.append((self.queue[floor]['downRequests'][0], floor, 'downRequests'))
                liveRequests = sorted(liveRequests, key=lambda x: x[0][-1])
                if len(liveRequests) == 0:
                    time.sleep(unitHaltDuration)
                    print('No request at T', ticks)
                    ticks += 1
                    continue
                else:
                    chosenRequest = liveRequests[0]
                    direction = chosenRequest[2]
                    goToFloor = chosenRequest[0][0]
                    peopleCount = chosenRequest[0][1]
                    requestTime = chosenRequest[0][2]
                    currentFloor = chosenRequest[1]
                    if direction == 'up':
                        self.state = 'movingup'
                    else:
                        self.state = 'movingdown'
                    self.currPersonCount += peopleCount
                    self.queue[currentFloor][direction].pop(0)
                    self.currFloor = currentFloor
                    print(f"Processing request {direction} from floor {currentFloor} to {goToFloor} at time {ticks}")

                    if self.state == 'movingup':
                        while self.currFloor < goToFloor:
                            self.currFloor += 1
                            self.recordMovingEnergy(currentFloor, self.currFloor, ticks, ticks + 1)
                            ticks += 1
                        self.state = 'idle'
                    else:
                        while self.currFloor > goToFloor:
                            self.currFloor -= 1
                            self.recordMovingEnergy(currentFloor, self.currFloor, ticks, ticks + 1)
                            ticks += 1
                        self.state = 'idle'

                    self.currPersonCount -= peopleCount
                    print(f"Arrived at floor {self.currFloor} at time {ticks}. Current person count: {self.currPersonCount}")
            time.sleep(unitHaltDuration)
            ticks += 1
        


class GenericLiftSystem:
    def __init__(self, lifts, systemLimit):
        self.lifts = lifts
        self.systemLimit = systemLimit
        self.requestsQueue = {}


    def addLift(self, lift):
        if len(self.lifts) < self.systemLimit:
            self.lifts.append(lift)
        else:
            print("System limit reached. Cannot add more lifts.")


    def distributeRequest(self, presentFloor, direction, goToFloor, peopleCount, timeOfRequest):
        availableLifts = [lift for lift in self.lifts if lift.state == 'idle' or lift.currPersonCount < lift.personLimit]
        if not availableLifts:
            print("No available lifts at the moment.")
            return

        chosenLift = min(availableLifts, key=lambda lift: abs(lift.currFloor - presentFloor))
        chosenLift.addRequest(presentFloor, direction, goToFloor, peopleCount, timeOfRequest)


    def runSystemSimulation(self, unitHaltDuration):
        for lift in self.lifts:
            lift.runSimulation(unitHaltDuration)



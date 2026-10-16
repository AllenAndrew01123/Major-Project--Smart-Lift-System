import time
import random
import numpy as np

class MLBasedLift:
    def __init__(self, floorLimit, personLimit, powerIdle, powerStandby, ratioIdle, ratioStandby, timeAverageTravel, numTrips, percentAvgDist, loadFactor, estRunningEnergy, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.floorLimit = floorLimit
        self.personLimit = personLimit
        self.currFloor = 0
        self.currPersonCount = 0
        self.queue = {floor: {'upRequests': [], 'downRequests': []} for floor in range(floorLimit + 1)}
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
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((floorLimit + 1, 2)) 
    

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


    def getState(self):
        return self.currFloor, self.currPersonCount, self.state


    def selectAction(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice([0, 1])  
        else:
            action = np.argmax(self.q_table[state[0], :]) 
        return action


    def takeAction(self, action):
        if action == 0:
            self.state = 'movingup'
            self.currFloor += 1
        else:
            self.state = 'movingdown'
            self.currFloor -= 1


    def updateQTable(self, state, action, reward, nextState):
        current_q = self.q_table[state[0], action]
        next_max_q = np.max(self.q_table[nextState[0], :]) 
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state[0], action] = new_q


    def getReward(self, state, action, nextState):
        reward = 0
        if self.state == 'idle' and self.currPersonCount > 0:
            reward += 10 
        elif self.state != 'idle' and abs(self.currFloor - nextState[0]) > 1:
            reward -= 5 
        return reward


    def runSimulation(self, unitHaltDuration):
        ticks = 0
        while self.queueHasRequests():
            if self.currPersonCount == 0:
                liveRequests = self.collectRequests(ticks)
                if len(liveRequests) == 0:
                    time.sleep(unitHaltDuration)
                    print(f'No request at T {ticks}')
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

                    while self.currFloor != goToFloor:
                        state = self.getState()
                        action = self.selectAction(state)
                        self.takeAction(action)
                        nextState = self.getState()
                        reward = self.getReward(state, action, nextState)
                        self.updateQTable(state, action, reward, nextState)
                        ticks += 1
                    self.state = 'idle'
                    self.currPersonCount -= peopleCount
                    print(f"Arrived at floor {self.currFloor} at time {ticks}. Current person count: {self.currPersonCount}")
            time.sleep(unitHaltDuration)
            ticks += 1


    def queueHasRequests(self):
        return any(len(self.queue[floor]['upRequests']) > 0 or len(self.queue[floor]['downRequests']) > 0 for floor in self.queue)


    def collectRequests(self, ticks):
        liveRequests = []
        for floor in range(0, self.floorLimit + 1):
            if len(self.queue[floor]['upRequests']) != 0 and self.queue[floor]['upRequests'][0][-1] <= ticks:
                liveRequests.append((self.queue[floor]['upRequests'][0], floor, 'upRequests'))
            if len(self.queue[floor]['downRequests']) != 0 and self.queue[floor]['downRequests'][0][-1] <= ticks:
                liveRequests.append((self.queue[floor]['downRequests'][0], floor, 'downRequests'))
        liveRequests = sorted(liveRequests, key=lambda x: x[0][-1])
        return liveRequests



class MLBasedLiftSystem:
    def __init__(self, lifts, systemLimit):
        self.lifts = lifts
        self.systemLimit = systemLimit


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

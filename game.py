import numpy


# States are couples (x,y), actions are couples in [(0,1), (1,0), (0,-1), (1,-1)]

def getBestAction(pos, objective):
    minimumD = numpy.Infinity
    minimumA = numpy.array([0,1])
    for action in [numpy.array([0,1]), numpy.array([1,0]), numpy.array([-1,0]), numpy.array([0,-1])]:
        distance = numpy.linalg.norm((pos + action - objective),2)
        if(distance <= minimumD):
            minimumA = action
            minimumD = distance
    return minimumA

def isInGrid(m,n,pos):
    if(pos[0]>=0 and pos[0]<=m-1 and pos[1]>=0 and pos[1]<=n-1):
        return True
    else:
        return False

def getTrajectory(m,n, policy):
    start = numpy.array([m//2, 2])
    objective = numpy.array([[numpy.random.randint(0,m), numpy.random.randint(0,n)]])
    states = numpy.array([start])
    count = 0
    while (states[-1]- objective).any() and count < 10:
        if(policy == None):
            action = getBestAction(states[-1], objective)
        else:
            action = policy(states[-1])
        if(isInGrid(m,n,states[-1]+action)):
            states = numpy.vstack((states,states[-1]+action))
        else:
            states = numpy.vstack((states,states[-1]))
        count += 1
    return states

def getActionSpace():
    return [numpy.array([0,1]), numpy.array([1,0]), numpy.array([-1,0]), numpy.array([0,-1])]

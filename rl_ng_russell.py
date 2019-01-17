import game
from scipy import optimize
import numpy

grid_lines = 5
grid_rows = 10
action_space = game.getActionSpace()
gamma = 0.95

def getTrajectories(policy,m=50):
    trajectories = []
    for i in range(m):
        trajectories.append(game.getTrajectory(grid_lines, grid_rows, policy))
    return trajectories

def find_alphas(sum_diff):
    c = sum_diff
    print(c)
    A = numpy.empty((0,d))
    for i in range(d):
        temp = numpy.zeros((2,d))
        temp[0,i]=1
        temp[1,i]=-1
        A = numpy.vstack((A,temp))
    return optimize.linprog(-c, A, numpy.ones((1,len(A)))).x

def findBestPolicy(alphas):
    return None



def computeValue(alphas, phis, policy, index):
    trajectories = getTrajectories(policy)
    s = 0
    for trajectory in trajectories:
        s += numpy.sum([gamma**u*phis[index](trajectory[u]) for u in range(len(trajectory))])
    return s/len(trajectories)

def computeValueArray(alphas,phis,policy):
    return numpy.array([computeValue(alphas, phis, policy,index) for index in range(d)])

def computeSumsForOptimization(alphas, phis, policies,value_max):
    sum_diff = numpy.array([])
    values = numpy.zeros((len(policies),d))
    for i in range(len(policies)):
        values[i] = computeValueArray(alphas, phis, policies[i])
    print(values)
    sum_diff = numpy.array([numpy.sum(numpy.array([value_max]*len(policies))-values[:,j]) for j in range(d)])
    return sum_diff


nbiter = 50
phis = []
for a in range(grid_lines)[::2]:
    for b in range(grid_rows)[::2]:
        phis.append(lambda c : numpy.exp(-((c[0]-a)**2+(c[1]-b)**2)))
d = len(phis)
print("Number of functions : "+str(d))

alphas = numpy.random.rand(1,d)
#On a R = sum(alpha[i]*phis[i]), le but est de trouver les alphas
policies = [lambda s : action_space[numpy.random.randint(0,len(action_space))]]
for k in range(nbiter):
    value_max = (computeValueArray(alphas,phis,None)).dot(alphas.reshape(1,d).T)[0]
    print("Value")
    print(value_max)
    print("\n")
    alphas = find_alphas(computeSumsForOptimization(alphas, phis, policies, value_max))
    print("New alphas : "+str(alphas))
    policies.append(findBestPolicy(alphas))

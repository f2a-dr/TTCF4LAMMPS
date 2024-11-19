import numpy as np

nSteps = 4
nBins = 3
nVariables = 2

B_pr = np.zeros([nSteps, nVariables, nBins])
B_gl = np.zeros([nSteps, nVariables, 1])

t = np.array([1])
print(np.concatenate((B_pr[0], t)))

print(B_pr[0])
# print(B_gl[0])

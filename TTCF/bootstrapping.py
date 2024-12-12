import numpy as np
import warnings


def resample(array, nResamples):
    batches = []
    arrayShape = list(array.shape)
    arrayShape.insert(0, nResamples)
    arrayShape = tuple(arrayShape)
    batches = np.zeros(arrayShape)
    for i in range(nResamples):
        newSample = np.random.randint(array.shape[-1], size=(array.shape[-1]))
        batches[i] = array[:,newSample]
    return batches


def averageWithin(array):
    return np.mean(array, axis=-1)


def averageBetween(array):
    return np.mean(array, axis=-1).T


def confidenceInterval(array, nResamples, intervalPercentage=95):
    if (0.005*nResamples) % 1 != 0:
        warnings.warn('It would be better to use a number of resamples that is a mutiple of 200')
    timesteps = array.shape[0]
    lowPercentile = int((100-intervalPercentage/2)*nResamples)
    highPercentile = int((100-(100-intervalPercentage/2))*nResamples)
    confInterval = np.zeros((timesteps, 2))
    for i in range(len(array)):
        sortedArray = np.sort(array[i])
        confInterval[i][0] = sortedArray[lowPercentile]
        confInterval[i][0] = sortedArray[highPercentile]

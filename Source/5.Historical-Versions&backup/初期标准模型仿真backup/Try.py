import numpy as np

HOLDING_TIME_LST = np.random.exponential(1, 60)
average_value = np.mean(HOLDING_TIME_LST)
print (HOLDING_TIME_LST)
print (average_value)
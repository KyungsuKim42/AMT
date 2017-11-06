import pdb
import numpy as np
import utils


data = 0.5*np.random.randn(100,7,264) + 1
pdb.set_trace()

data = utils.standardize(data,2)

print data

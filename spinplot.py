import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        a = np.loadtxt( arg )
        print( arg + ' - ' + str(a.shape) )

        plt.imshow(a)
        plt.show()


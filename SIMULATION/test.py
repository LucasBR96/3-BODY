from simmu import simmu
from synthesis import simulate
from visualization import plot_simulation

import matplotlib
import matplotlib.pyplot as plt
import numpy as np  
if __name__ == "__main__":

    vel_0 = np.zeros( ( 3 , 2 ) )
    vel_0[ 0 , 0 ] = -.1

    S0 = simmu(  manual_setting = True , vel_0 = vel_0.copy(), run_time = 15 )
    simulate( 0 , Sm = S0 )
    ax_0 = plot_simulation( 0 )

    S1 = simmu( manual_setting = True , run_time = 15 )
    simulate( 1 , Sm = S1 )
    ax_1 = plot_simulation( 1 )

    # img , axs = plt.subplots( 1 , 2 )
    # axs[ 0 ] = ax_0
    # axs[ 1 ] = ax_1
    plt.show()
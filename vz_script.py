from SIMULATION.visualization import plot_simulation
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":

    arglst = sys.argv
    arglst.append( '0' )

    try:
        n = int( arglst[ 1 ] )
    except ValueError:
        s = f"the value {arglst[1]} could not be casted, setting to 0."
        print( s )
        n = 0
    
    plot_simulation( n )
    plt.show()
    

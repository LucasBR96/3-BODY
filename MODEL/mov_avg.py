from collections import deque

class mob_mean_gen:

    def __init__( self , horizon = 100 ):
        
        self.horizon = horizon
        self.mean = 0
        self.buff = deque(list())
    
    def __call__(self, val) -> float:
        
        self.buff.append( val )
        n = len( self.buff ) - 1

        if n == 0:
            self.mean = val
        elif n < self.horizon:
            self.mean = ( self.mean*n + val )/( n + 1 )
        else:
            bottom = self.buff.popleft()
            self.mean += ( val - bottom )/n
        
        return self.mean
    
    def clear( self ):
        self.buff.clear()

if __name__ == "__main__":

    import numpy as np
    import random as rd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    

    base : np.ndarray = np.random.random( 2000 ) -.5
    base = base.cumsum()

    plt.plot( range( 2000 ) , base , "-k" , label = "original" )

    hs = [ 5 , 10 , 50 , 100 ]
    idx = np.linspace( 0 , 1 , 4 )
    for i in range( 4 ):
        mob = mob_mean_gen( hs[i] )
        nu_base = np.zeros( 2000 )
        for j in range( 2000 ):
            nu_base[ j ] = mob( base[ j ] )
        plt.plot( nu_base , linestyle = "--" , label = f"h = {hs[i]}")
    plt.legend()
    plt.show()


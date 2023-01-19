import numpy as np

class simmu:

    MIN_GRAN = 10
    MAX_GRAN = 100

    MIN_ITER = 2048
    MAX_ITER = 4096

    def __init__( self , gran = 10 , max_iter = 2048 ):
        
        #----------------------------------------------
        # subdivisions of one second
        self.gran = np.clip(
            gran,
            simmu.MIN_GRAN,
            simmu.MAX_GRAN
        )

        #----------------------------------------------
        # maximum number of iterations before the simulation
        # ends
        self.max_iter = np.clip(
            max_iter,
            simmu.MIN_ITER,
            simmu.MAX_ITER
        )

        self.iteration = 0
        self.pos = np.zeros( ( 3 , 2 ) )
        self.vel = np.zeros( ( 3 , 2 ) )
    
    def __call__( self ):
        
        if self.iteration == 0:
            # code here
            pass

        elif self.iteration < self.max_iter:
            # code here
            pass

        self.iteration += 1
        tup = (
            self.pos.copy(),
            self.vel.copy()
        )
        return tup


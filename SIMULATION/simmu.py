import numpy as np

class simmu:

    MIN_GRAN = 10
    MAX_GRAN = 100

    MIN_ITER = 2048
    MAX_ITER = 4096

    RADIUS = .1
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
            self.init_planets()
            pass

        elif self.iteration < self.max_iter:
            
            t1 = 1/self.gran
            t2 = ( t1**2 )/2
            for _ in range( self.gran ):

                acc = self.get_acc()
                self.pos += self.vel*t1 + acc*t2
                self.vel += acc*t1
            
        self.iteration += 1
        tup = (
            self.iteration - 1,
            self.pos.copy(),
            self.vel.copy()
        )
        return tup

    def init_planets( self ):

        #---------------------------------------
        # planet 1 starts in the same position
        # for any simulation 
        self.pos[ 0 ] = np.array([ 1. , 0 ])

        #---------------------------------------
        # planet 2 spawns in the second quadrant
        # of the trigonometric cycle, with unitary
        # distance to the origin
        theta = 0.5*np.pi*( 1 + np.random.random() )
        self.pos[ 1 ] = np.array([
            np.cos( theta ),
            np.sin( theta )
        ])

        #----------------------------------------
        # the center of mass between the three
        # planets is the origin on spawn. The starting
        # position of the 3rd planet is deduced from
        # the previous 2
        self.pos[ 2 ] = -( self.pos[ 1 ] + self.pos[ 0 ] )
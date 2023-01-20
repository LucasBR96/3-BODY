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
        self.pos : np.ndarray = np.zeros( ( 3 , 2 ) )
        self.vel : np.ndarray = np.zeros( ( 3 , 2 ) )
    
    def __call__( self ):
        
        #-------------------------------------
        # Here the simulation haven't even begun.
        # Initialize planets positions and return
        # the starting values
        if self.iteration == 0:
            self.init_planets()
            pass
        
        #----------------------------------------
        # Elapse one second of motion using the 
        # basic gravitational equations and iterate
        # using the rung kutta method.
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
    
    def get_acc( self ):

        xs : np.ndarray = self.pos[ : , 0 ]
        dx = xs.reshape( ( 1 , 3 ) ) - xs.reshape( ( 3 , 1 ) )
        dx = remove_diagonals( dx )

        ys : np.ndarray = self.pos[ : , 1 ]
        dy = ys.reshape( ( 1 , 3 ) ) - ys.reshape( ( 3 , 1 ) )
        dy = remove_diagonals( dx )

        dist_sqr = dx**2 + dy**2
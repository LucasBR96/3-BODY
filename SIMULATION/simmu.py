import numpy as np

def rmv_diagonals( mat : np.ndarray ) -> np.ndarray:
    
    '''
    Removes the main diagonal of a 2d square matrix.
    This implementation only works for 3 x 3 matrices,
    given the scope of this project.

    Example:

    >>> mat = np.array([
        [ 3 , 2. , 0 ],
        [ 5 , 5 , 1. ],
        [ 1. , 0 , 0 ]
    ])
    >>> rmv_diagonals( mat )
    array([[2., 0.],
           [5., 1.],
           [1., 0.]])
    '''

    shp = mat.shape
    if shp != ( 3 , 3 ):
        raise ValueError( f"shape must be (3, 3) but given array has shape of {shp}")

    new_mat = np.zeros( ( 3 , 2 ) )
    new_mat[ 0 , 0 ] = mat[ 0 , 1 ]
    new_mat[ 0 , 1 ] = mat[ 0 , 2 ]
    new_mat[ 1 , 0 ] = mat[ 1 , 0 ]
    new_mat[ 1 , 1 ] = mat[ 1 , 2 ]
    new_mat[ 2 , 0 ] = mat[ 2 , 0 ]
    new_mat[ 2 , 1 ] = mat[ 2 , 1 ]
    return new_mat

class simmu:

    MIN_GRAN = 10
    MAX_GRAN = 100

    MIN_ITER = 2048
    MAX_ITER = 4096

    RADIUS = .1
    MAX_ACCL = ( 2*RADIUS )**( -2 )

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
        dx = rmv_diagonals( dx )

        ys : np.ndarray = self.pos[ : , 1 ]
        dy = ys.reshape( ( 1 , 3 ) ) - ys.reshape( ( 3 , 1 ) )
        dy = rmv_diagonals( dx )

        dist_sqr = dx**2 + dy**2

        #--------------------------------------
        # finding the modular acceleration. For
        # numeric reasons, if the square distance
        # is less than the squared sum of two planet
        # radius, the aceleration is capped.
        mod_acc = np.where(
            dist_sqr > simmu.MAX_ACCL**( -1 ),
            dist_sqr**( -1 ),
            simmu.MAX_ACCL
        )

        dist = np.sqrt( dist_sqr )
        ax   = mod_acc*( -dx/dist )
        ay   = mod_acc*( -dy/dist )

        acc = np.zeros( ( 3 , 2 ) )
        acc[ : , 0 ] = ax.sum( axis = 1 )
        acc[ : , 1 ] = ay.sum( axis = 1 )
        return acc
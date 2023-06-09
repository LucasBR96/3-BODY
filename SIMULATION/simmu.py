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


class tstep_adapter:
    
    def __init__( self , t_base , acc = None ):

        self.t_base = t_base
        self.acc = acc
    
    def __call__( self , new_acc ):

        if new_acc:
            old_acc , self.acc = self.acc , new_acc
            if not( old_acc is None):
                old_t = self.t_base
                self.t_base = np.sqrt( old_acc/new_acc )*old_t
        return self.t_base

class simmu:

    MIN_GRAN = 1e-4
    MAX_GRAN = 1e-1

    MIN_ITER = 1
    MAX_ITER = 10**3

    RADIUS = 0.1
    MIN_DIST = ( 2*RADIUS )**( 2 )

    POS_0 = np.array([
        [ 1. , 0. ],
        [ -1. , 1 ],
        [ 0 , -1. ]
    ])

    VEL_0 = np.zeros( ( 3 , 2 ) )

    def __init__( self , h_step = .02, r_step = .25 , run_time = 10.
    , adaptative = True , manual_setting = False ,
    pos_0 = None , vel_0 = None ):
        
        #----------------------------------------------
        # time-step size
        self.t1 = np.clip(
            h_step,
            simmu.MIN_GRAN,
            simmu.MAX_GRAN
        )
        self.t2 = .5*( self.t1**2 )

        #---------------------------------------------
        # recording step size
        self.r_step = r_step

        #------------------------------------------------
        # adptation settings
        self.adaptative = adaptative
        self.t_adpt = None
        if adaptative:
            self.t_adpt = tstep_adapter( self.t1 )

        #----------------------------------------------
        # maximum number of iterations before the simulation
        # ends
        self.run_time = np.clip(
            run_time,
            simmu.MIN_ITER,
            simmu.MAX_ITER
        )
        self.curr_time = 0
        
        #-------------------------------------------------
        # Setting up initial positions and velocities
        self.manual_setting = manual_setting
        if not manual_setting:
            pos_0 = np.zeros( ( 3 , 2 ) )
            vel_0 = np.zeros( ( 3 , 2 ) )
        else:
            if pos_0 is None:
                pos_0 = simmu.POS_0.copy()
            if vel_0 is None:
                vel_0 = simmu.VEL_0.copy()
        self.pos = pos_0
        self.vel = vel_0
    
    def __call__( self ):
        
        #-------------------------------------
        # Here the simulation haven't even begun.
        # Initialize planets positions and return
        # the starting values
        if not( self.curr_time or self.manual_setting ):
            self.init_planets()
            pass
        
        #----------------------------------------
        # Elapse one second of motion using the 
        # basic gravitational equations and iterate
        # using the rung kutta method.
        elif self.curr_time < self.run_time:
            self._update_system()
            
        tup = (
            self.curr_time,
            self.pos.copy(),
            self.vel.copy()
        )
        self.curr_time += self.r_step
        
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

        #------------------------------------------
        # All velocities are random, with module equal to
        # 1. The linear momentum must be equal to 0 , 0 
        theta = 2*np.pi*np.random.random( 3 )
        self.vel[ :3 , 0 ] = np.cos( theta )
        self.vel[ :3 , 1 ] = np.sin( theta )
        
    
    def _update_system( self ):

        base = 0
        while base < self.r_step:

            acc = self.get_acc()
            if self.adaptative:
                t_left = self.r_step - base
                self.update_tstep( acc , t_left )

            self.pos += self.vel*self.t1 + acc*self.t2
            self.vel += acc*self.t1

            base += self.t1

    def get_acc( self ):

        xs : np.ndarray = self.pos[ : , 0 ]
        dx = xs.reshape( ( 1 , 3 ) ) - xs.reshape( ( 3 , 1 ) )
        dx = rmv_diagonals( dx )

        ys : np.ndarray = self.pos[ : , 1 ]
        dy = ys.reshape( ( 1 , 3 ) ) - ys.reshape( ( 3 , 1 ) )
        dy = rmv_diagonals( dy )

        dist_sqr = dx**2 + dy**2

        #--------------------------------------
        # finding the modular acceleration. For
        # numeric reasons, if the square distance
        # is less than the squared sum of two planet
        # radius, the aceleration is capped.
        mod_acc = 1/np.clip(
            dist_sqr,
            simmu.MIN_DIST,
            None
        )
        
        dist = np.sqrt( dist_sqr )
        ax   = mod_acc*( dx/dist )
        ay   = mod_acc*( dy/dist )

        acc = np.zeros( ( 3 , 2 ) )
        acc[ : , 0 ] = ax.sum( axis = 1 )
        acc[ : , 1 ] = ay.sum( axis = 1 )
        return acc
    
    def update_tstep( self , acc , t_left ):
        
        acc_squared = ( acc**2 ).sum( axis = 1 )
        acc_mod     = np.sqrt( acc_squared )
        max_acc     = acc_mod.max()
        
        t1 = min( self.t_adpt( max_acc ) , t_left )
        self.t1 = t1
        self.t2 = .5*( self.t1**2 )
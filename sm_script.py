import SIMULATION.simmu as sm
import SIMULATION.synthesis as sy
import numpy as np

# def rnd_start():

#     pos = np.zeros( ( 3 , 2 ) )

#     theta1 = np.random.rand()*( np.pi/2 )
#     pos[ 0 ] = np.array( [ np.cos( theta1 ) , np.sin( theta1 ) ] )

#     theta2 = theta1 + ( 1 + np.random.rand() )*( np.pi/2 )
#     pos[ 1 ] = np.array( [ np.cos( theta2 ) , np.sin( theta2 ) ] )

#     pos[ 2 ] = -( pos[ 0 ] + pos[ 1 ] )

#     return pos

if __name__ == "__main__":

    h_step = .5*1e-3
    total_time = 120.

    for i in range( 400 , 410 ):
        
        Sm = sm.simmu(
            h_step = h_step,
            run_time = total_time
        )

        sy.simulate(
            i,
            Sm = Sm,
            verbose = False 
        )

import SIMULATION.simmu as sm
import SIMULATION.synthesis as sy

if __name__ == "__main__":

    h_step = 1e-3
    total_time = 120.

    for i in range( 25 , 350 ):
        
        Sm = sm.simmu(
            h_step = h_step,
            run_time = total_time
        )

        sy.simulate(
            i,
            Sm = Sm,
            verbose = False 
        )

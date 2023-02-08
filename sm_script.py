import SIMULATION.simmu as sm
import SIMULATION.synthesis as sy

if __name__ == "__main__":

    h_step = 1e-3
    total_time = 60.
    r_step = 2*1e-2

    for i in range( 100 ):
        
        Sm = sm.simmu(
            h_step = h_step,
            run_time = total_time
        )

        sy.simulate(
            i,
            Sm = Sm,
            step_size = r_step,
            verbose = False 
        )

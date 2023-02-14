import time
from typing import *

class clock:

    def __init__(self, max_time = 8, format : Literal[ 'seconds' , 'minutes' , 'hours' ] = 'hours' ):
        
        self.format = format
        self.base = 0

        k = 1
        if format == 'minutes':
            k = 60
        elif format == "hours":
            k = 3600
        self.max_time = max_time*k

    def tick( self ):
        
        def foo( fun ):
            def bar( *args ):
                ti = time.time()
                fun( *args )
                tj = time.time()
                self.base += tj - ti
            return bar
        return foo

    def is_done( self ):
        return self.base >= self.max_time
    
    def __str__( self ):
        
        base = self.base

        num_min , num_secs = base//60 , base%60
        num_hrs , num_min = num_min//60 , num_min%60

        secs_str = f"{num_secs:.4f}".rjust( 6 , "0")
        min_str = f"{num_min}".rjust( 2 , "0" )
        hrs_str = f"{num_hrs}".rjust( 2 , "0" )

        return f"{hrs_str}:{min_str}:{secs_str}"
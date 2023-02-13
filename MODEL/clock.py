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

            ti = time.time()
            fun()
            tj = time.time()
            self.base += tj - ti

        return foo

    def is_done( self ):
        return self.base >= self.max_time
    
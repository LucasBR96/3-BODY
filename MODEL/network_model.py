import torch as tc
import torch.utils as ut
import torch.nn as tnn

NUM_LAY = 9
LAY_SIZE = 575

class perceptron( tnn.Module ):

    def __init__( self , in_size = LAY_SIZE , out_size = LAY_SIZE ):

        super().__init__()
        self.lin = tnn.Linear( in_size , out_size )
        self.rel = tnn.Softplus( )
    
    def forward( self , X ):

        X = self.lin( X )
        return self.rel( X )
    

class stellar_model( tnn.Module ):

    def __init__( self , dim , inner_size = LAY_SIZE , num_lay = NUM_LAY ):
        super().__init__()

        self.dim = dim
        self.edge_size = dim[ 0 ]*dim[ 1 ]

        self.head = perceptron( self.edge_size , inner_size )
        self.body = tnn.Sequential(
            *[ perceptron( inner_size , inner_size ) for _ in range( num_lay ) ]
        )
        self.tail = tnn.Linear( inner_size , self.edge_size )
    
    def forward( self , S : tc.Tensor ):

        #-------------------------------------
        # unitary sample
        usamp : bool = ( S.ndim == 2 )
        if usamp:
            S = S.unsqueeze( 0 )
        
        #------------------------------
        # Feeding Through the network
        X : tc.Tensor
        X = S.flatten( 1 )

        y = tc.Tensor
        y = self.head( X )
        y = self.body( y )
        y = self.tail( y )

        #---------------------------------------
        # shaping into predicted state
        d0 , d1 = self.dim
        n = y.size()[ 0 ]
        S_prime = y.reshape( n , d0 , d1 )

        if usamp:
            S_prime = S_prime.squeeze()
        return S_prime
    

if __name__ == "__main__":

    device = "cuda" if tc.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    X = tc.rand( 2 , 5 , 4 ).to( device )
    print( f"Before P: \n \n {X[0]}")

    P = stellar_model( ( 5 , 4 ) , 10 , 2 ).to( device )
    X = P( X )

    print( f"\nAfter P: \n \n {X[ 0 ]}")
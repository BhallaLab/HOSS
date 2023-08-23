import heapq

class Row():
    data = []
    def __init__( self, idx ):
        self.idx = list(idx)
        self.tot = 0.0
        for dd, ii in zip( self.data, idx ):
            self.tot += dd[ ii ]

    def print( self ):
        for ii, dd in zip( self.idx, self.data ):
            print( "{:8.1f}".format( dd[ii] ), end = "" )
        print( "{:8.1f}".format( self.tot ) )

    def __lt__( self, other ):
        if self.tot < other.tot:
            return True
        elif self.tot == other.tot:
            return self.idx < other.idx
        return False

    def __eq__( self, other ):
        return self.idx == other.idx

def main():
    a = sorted([1, 1.2, 1.5, 2, 2.1, 4, 5, 5.5 ])
    b = sorted([10, 10.1, 10.4, 10.8, 12.1, 13, 15, 15.5 ])
    c = sorted([100, 100.5, 100.6, 101.8, 102.1, 102.3, 104, 107 ])
    Row.data = [a, b, c]


    vec = [Row( [0, 0, 0] )]
    heapq.heapify( vec )

    svec = []

    aii = 0
    bii = 0
    cii = 0

    while( len( svec ) < 20 ):
        print( len( vec ), len( svec ) )
        rr = heapq.heappop( vec )
        if len( vec ) > 0 and not rr == vec[0]:
            svec.append( rr )
        for ii in range( len( Row.data ) ):
            idx2 = list(rr.idx)
            idx2a = list(idx2)
            idx2[ii] += 1
            print( idx2a, idx2, ii )
            heapq.heappush( vec, Row( idx2 ) )


    for vv in sorted(svec):
        vv.print()

if __name__ == "__main__":
    main()

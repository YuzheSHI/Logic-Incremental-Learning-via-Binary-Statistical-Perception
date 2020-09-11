less(0,1).
less(1,2).
less(2,3).
less(3,4).
less(4,5).
less(5,6).
less(6,7).
less(7,8).
less(8,9).

smaller(A,B):-less(A,B).
smaller(A,B):-less(A,C),smaller(C,B).

sorted([A|[B|_]]):-smaller(A,B).
sorted([A|[B|C]]):-smaller(A,B),sorted([B|C]).


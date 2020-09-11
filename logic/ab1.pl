:-use_module('metagol').

% use background knowledge
body_pred(smaller/2).
body_pred(less/2).
body_pred(sorted/1).

% metarules
metarule([P,Q,A,B],[P,[A|[B|_]]],[[Q,A,B]]).
metarule([P,Q,A,B,C],[P,[A|[B|C]]],[[Q,A,B],P,[B|C]]).

% background knowledge 
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

a:-
    Pos = [
        t([0,1]),
        t([4,7]),
        t([0,1,2]),
        t([4,6,9]),
        t([1,2,3,4,5]),
        t([1,3,5,7,9]),
        t([1,2,4,6,7,9])
    ],
    Neg = [
        t([1,0]),
        t([7,3]),
        t([5,3,2]),
        t([1,2,6,4,5]),
        t([1,3,8,6,9]),
        t([1,2,4,5,6,8,7]),
        t([0,1,2,4,6,7,9,8])
    ],
    learn(Pos,Neg).

:-time(a).
:-use_module('metagol').

body_pred(step/2).

metarule(
    [P, Q, R],
    [P, A, B],
    [[Q, A, B],[R, A, B]]
).

step(A, B) :-
    succ(A, B).

a :-
    Pos = [
        one(0, 1, 1)
    ],
    Neg = [],
    learn(Pos, Neg, Prog),
    ppring(Prog).

:- time(a).
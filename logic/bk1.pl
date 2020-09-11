one(1).
digit(1):-one(1).

smaller_succ(digit(0),digit(1)).
smaller_succ(digit(1),digit(2)).
smaller_succ(digit(2),digit(3)).
smaller_succ(digit(3),digit(4)).
smaller_succ(digit(4),digit(5)).
smaller_succ(digit(5),digit(6)).
smaller_succ(digit(6),digit(7)).
smaller_succ(digit(7),digit(8)).
smaller_succ(digit(8),digit(9)).

greater_pref(A,B):-smaller_succ(B,A).

smaller(A,B):-smaller_succ(A,B).
smaller(A,B):-smaller_succ(A,C),smaller(C,B).

greater(A,B):-greater_pref(A,B).
greater(A,B):-greater_pref(A,C),greater(C,B).

equal(A,B):-(\+smaller(A,B)),\+greater(A,B).
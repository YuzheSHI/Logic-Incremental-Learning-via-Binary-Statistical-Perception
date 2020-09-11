func_test(Atom1, Atom2, Condition):-
    Atom1 = [P, X, Z],
    Atom2 = [P, X, T],
    Condition = (Z = T).
% initial background knowledge
is_least(0).

inverse(A, R) :-
    R is (-A).

difference(A, B, R) :-
    (B > A) -> inverse(A, T),
    plus(B, T, R).
difference(A, B, R) :-
    (B < A) -> inverse(B, T),
    plus(A, T, R).

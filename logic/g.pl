% The production for A, B in Z_10 is addition-modular-10 and it follows closure
% A p B = (A + B) mod 10
add(A, B, R) :-
    !, plus(A, B, R).
add(A, B, R) :-
    plus(A, B, T),
    (T > 10) -> plus(T, -10, R).


% The elements of Z_10 follows the asscociativity
% (A p B) p C = A p (B p C)


% The identity element of Z_10 is zero
% A p 0 = 0 p A = A


% The inverse element for A in Z_10 is (10 - A)
% A p 0 = 0 p A = 1
abs(X, R) :- 
    number(X),
    (X < 0) -> R is (-X).
inverse(A,R) :- 
    add(A, -10, T),
    abs(T, R).


:-use_module('metagol').
metagol:unfold_program.
% digits 0~9 are elements of a modular-10 group Z_10


% The production for A, B in Z_10 is addition-modular-10 and it follows closure
% A p B = (A + B) mod 10


body_pred(plus/3).


metarule([P,Q],[P,A,B,C],[[Q,A,B,C]]).
metarule([P,Q],[P,A,B,_C],[[Q,A,B,_D]]).

production:-
    Pos = [
        t(1,2,3),
        t(2,5,7),
        t(1,3,4),
        t(1,5,6),
        t(2,4,6),
        t(3,6,9),
        t(4,4,8),
        t(4,5,9),
        t(3,5,8),
        t(6,2,8),
        t(6,3,9),
        t(7,2,9),
        t(1,9,0),
        t(2,9,1),
        t(2,8,0),
        t(3,7,0),
        t(3,8,1),
        t(4,8,2),
        t(4,9,3),
        t(5,9,4),
        t(6,8,4),
        t(9,9,8)
    ],
    Neg = [
        t(1,9,10),
        t(2,9,11),
        t(2,8,10),
        t(3,7,10),
        t(3,8,11),
        t(4,8,12),
        t(4,9,13),
        t(5,9,14),
        t(6,8,14),
        t(9,9,18)
    ],
    learn(Pos,Neg).

% The elements of Z_10 follows the asscociativity
% (A p B) p C = A p (B p C)


% The identity element of Z_10 is zero
% A p 0 = 0 p A = A


% The inverse element for A in Z_10 is (10 - A)
% A p 0 = 0 p A = 1


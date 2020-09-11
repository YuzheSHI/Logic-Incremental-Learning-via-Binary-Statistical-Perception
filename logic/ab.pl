:-use_module('metagol').

% use background knowledge
%body_pred(digit/1).
body_pred(smaller_succ/2).
%body_pred(greater_pref/2).
%body_pred(smaller/2).
%body_pred(greater/2).
%prim(smaller_succ/2).
%prim(digit/1).

% metarules
metarule([P,Q,A],[P,_A],[[Q,A,_C]]).
metarule([R,Q,A],[R,_A],[[Q,_C,A]]).
%metarule(chain,[P,Q,A,B],[P,A,B],[[Q,A,C],[Q,C,B]]).


% background knowledge
digit(0).
digit(1).

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

% training
a:-
   Pos = [
        t(1),
        t(2)
   ],
   Neg = [

   ],
   learn(Pos,Neg). 

:-time(a).
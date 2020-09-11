:-use_module('metagol').

metagol: functional.

func_test(Atom1, Atom2, Condition):-
    Atom1 = [P, X, Y], 
    Atom2 = [P, X, Z],
    Condition = (Y = Z).

metarule(
    identity, 
    [P, Q],
    [P, X, Y],
    [[Q, X, Y]]
).

% metarule(
%     chain, 
%     [P, Q, R],
%     [P, X, Y],
%     [[Q, X, Z], [R, Z, Y]]  
% ).

% metarule(
%     inverse, 
%     [P, Q],
%     [P, X, Y],
%     [[Q, Y, X]]
% ).

metarule(
    tailrec, 
    [P, Q],
    [P, X, Y],
    [[Q, X, Z], [P, Z, Y]]    
).

metarule(
    headrec,
    [P, Q],
    [P, X, Y],
    [[P, X, Z], [Q, Z, Y]]
).


body_pred(succ/2).


a :-
    Episode_1 = [
        diff_1(0, 1)
    ]/[
        diff_1(1, 0)
    ],

    Episode_2 = [
        diff_2(0, 2)
    ]/[],

    Episode_3 = [
        diff_3(0, 3)
    ]/[],

    Episode_4 = [
        diff_4(0, 4)
    ]/[],

    Episode_5 = [
        diff_5(0, 5)
    ]/[],

    Episode_6 = [
        diff_6(0, 6)
    ]/[],

    Episode_7 = [
        diff_7(0, 7)
    ]/[],

    Episode_8 = [
        diff_8(0, 8)
    ]/[],

    Episode_9 = [
        diff_9(0, 9)
    ]/[],

    learn_seq([Episode_1, Episode_2, Episode_3, Episode_4, Episode_5, Episode_6, Episode_7, Episode_8, Episode_9], Prog),
    pprint(Prog).


b :- 
    Episode_1 = [
    diff_1(0, 1)
    ]/[],

    Episode_2 = [
    diff_2(0, 2)
    ]/[],

    Episode_3 = [
    diff_3(0, 3)
    ]/[],

    Episode_4 = [
    diff_4(0, 4)
    ]/[],

    Episode_5 = [
    diff_5(0, 5)
    ]/[],

    Episode_6 = [
    diff_6(0, 6)
    ]/[],

    Episode_7 = [
    diff_7(0, 7)
    ]/[],

    Episode_8 = [
    diff_8(0, 8)
    ]/[],

    Episode_9 = [
    diff_9(0, 9)
    ]/[],

    learn_seq([Episode_2, Episode_3, Episode_4, Episode_5, Episode_6, Episode_7, Episode_8, Episode_9, Episode_1], Prog),
    pprint(Prog).

c :- 
    Episode_1 = [
    diff_1(0, 1)
    ]/[],

    Episode_2 = [
    diff_2(0, 2)
    ]/[],

    Episode_3 = [
    diff_3(0, 3)
    ]/[],

    Episode_4 = [
    diff_4(0, 4)
    ]/[],

    Episode_5 = [
    diff_5(0, 5)
    ]/[],

    Episode_6 = [
    diff_6(0, 6)
    ]/[],

    Episode_7 = [
    diff_7(0, 7)
    ]/[],

    Episode_8 = [
    diff_8(0, 8)
    ]/[],

    Episode_9 = [
    diff_9(0, 9)
    ]/[],

    learn_seq([Episode_3, Episode_4, Episode_5, Episode_6, Episode_7, Episode_8, Episode_9, Episode_1, Episode_2], Prog),
    pprint(Prog).


d :-  
    Episode_1 = [
    diff_1(0, 1)
    ]/[],

    Episode_2 = [
    diff_2(0, 2)
    ]/[],

    Episode_3 = [
    diff_3(0, 3)
    ]/[],

    Episode_4 = [
    diff_4(0, 4)
    ]/[],

    Episode_5 = [
    diff_5(0, 5)
    ]/[],

    Episode_6 = [
    diff_6(0, 6)
    ]/[],

    Episode_7 = [
    diff_7(0, 7)
    ]/[],

    Episode_8 = [
    diff_8(0, 8)
    ]/[],

    Episode_9 = [
    diff_9(0, 9)
    ]/[],

    learn_seq([Episode_4, Episode_5, Episode_6, Episode_7, Episode_8, Episode_9, Episode_1, Episode_2, Episode_3], Prog),
    pprint(Prog).


e :- 
    Episode_1 = [
    diff_1(0, 1)
    ]/[],

    Episode_2 = [
    diff_2(0, 2)
    ]/[],

    Episode_3 = [
    diff_3(0, 3)
    ]/[],

    Episode_4 = [
    diff_4(0, 4)
    ]/[],

    Episode_5 = [
    diff_5(0, 5)
    ]/[],

    Episode_6 = [
    diff_6(0, 6)
    ]/[],

    Episode_7 = [
    diff_7(0, 7)
    ]/[],

    Episode_8 = [
    diff_8(0, 8)
    ]/[],

    Episode_9 = [
    diff_9(0, 9)
    ]/[],

    learn_seq([Episode_5, Episode_6, Episode_7, Episode_8, Episode_9, Episode_1, Episode_2, Episode_3, Episode_4], Prog),
    pprint(Prog).


f :- 
    Episode_1 = [
    diff_1(0, 1)
    ]/[],

    Episode_2 = [
    diff_2(0, 2)
    ]/[],

    Episode_3 = [
    diff_3(0, 3)
    ]/[],

    Episode_4 = [
    diff_4(0, 4)
    ]/[],

    Episode_5 = [
    diff_5(0, 5)
    ]/[],

    Episode_6 = [
    diff_6(0, 6)
    ]/[],

    Episode_7 = [
    diff_7(0, 7)
    ]/[],

    Episode_8 = [
    diff_8(0, 8)
    ]/[],

    Episode_9 = [
    diff_9(0, 9)
    ]/[],

    learn_seq([Episode_6, Episode_7, Episode_8, Episode_9, Episode_1, Episode_2, Episode_3, Episode_4, Episode_5], Prog),
    pprint(Prog).


g :- 
    Episode_1 = [
    diff_1(0, 1)
    ]/[],

    Episode_2 = [
    diff_2(0, 2)
    ]/[],

    Episode_3 = [
    diff_3(0, 3)
    ]/[],

    Episode_4 = [
    diff_4(0, 4)
    ]/[],

    Episode_5 = [
    diff_5(0, 5)
    ]/[],

    Episode_6 = [
    diff_6(0, 6)
    ]/[],

    Episode_7 = [
    diff_7(0, 7)
    ]/[],

    Episode_8 = [
    diff_8(0, 8)
    ]/[],

    Episode_9 = [
    diff_9(0, 9)
    ]/[],

    learn_seq([Episode_7, Episode_8, Episode_9, Episode_1, Episode_2, Episode_3, Episode_4, Episode_5, Episode_6], Prog),
    pprint(Prog).


h:- 
    Episode_1 = [
    diff_1(0, 1)
    ]/[],

    Episode_2 = [
    diff_2(0, 2)
    ]/[],

    Episode_3 = [
    diff_3(0, 3)
    ]/[],

    Episode_4 = [
    diff_4(0, 4)
    ]/[],

    Episode_5 = [
    diff_5(0, 5)
    ]/[],

    Episode_6 = [
    diff_6(0, 6)
    ]/[],

    Episode_7 = [
    diff_7(0, 7)
    ]/[],

    Episode_8 = [
    diff_8(0, 8)
    ]/[],

    Episode_9 = [
    diff_9(0, 9)
    ]/[],

    learn_seq([Episode_8, Episode_9, Episode_1, Episode_2, Episode_3, Episode_4, Episode_5, Episode_6, Episode_7], Prog),
    pprint(Prog).


i :- 
    Episode_1 = [
    diff_1(0, 1)
    ]/[],

    Episode_2 = [
    diff_2(0, 2)
    ]/[],

    Episode_3 = [
    diff_3(0, 3)
    ]/[],

    Episode_4 = [
    diff_4(0, 4)
    ]/[],

    Episode_5 = [
    diff_5(0, 5)
    ]/[],

    Episode_6 = [
    diff_6(0, 6)
    ]/[],

    Episode_7 = [
    diff_7(0, 7)
    ]/[],

    Episode_8 = [
    diff_8(0, 8)
    ]/[],

    Episode_9 = [
    diff_9(0, 9)
    ]/[],

    learn_seq([Episode_9, Episode_1, Episode_2, Episode_3, Episode_4, Episode_5, Episode_6, Episode_7, Episode_8], Prog),
    pprint(Prog).

j :- 
    Episode_1 = [
    diff_1(0, 1)
    ]/[],

    Episode_2 = [
    diff_2(0, 2)
    ]/[],

    Episode_3 = [
    diff_3(0, 3)
    ]/[],

    Episode_4 = [
    diff_4(0, 4)
    ]/[],

    Episode_5 = [
    diff_5(0, 5)
    ]/[],

    Episode_6 = [
    diff_6(0, 6)
    ]/[],

    Episode_7 = [
    diff_7(0, 7)
    ]/[],

    Episode_8 = [
    diff_8(0, 8)
    ]/[],

    Episode_9 = [
    diff_9(0, 9)
    ]/[],

    learn_seq([Episode_9, Episode_8, Episode_7, Episode_6,Episode_5, Episode_4, Episode_3, Episode_2, Episode_1], Prog),
    pprint(Prog).

k :- 
    Episode_1 = [
    diff_1(0, 1)
    ]/[],

    Episode_2 = [
    diff_2(0, 2)
    ]/[],

    Episode_3 = [
    diff_3(0, 3)
    ]/[],

    Episode_4 = [
    diff_4(0, 4)
    ]/[],

    Episode_5 = [
    diff_5(0, 5)
    ]/[],

    Episode_6 = [
    diff_6(0, 6)
    ]/[],

    Episode_7 = [
    diff_7(0, 7)
    ]/[],

    Episode_8 = [
    diff_8(0, 8)
    ]/[],

    Episode_9 = [
    diff_9(0, 9)
    ]/[],

    learn_seq([Episode_9, Episode_1, Episode_3, Episode_2,  Episode_4, Episode_7, Episode_5, Episode_6, Episode_8], Prog),
    pprint(Prog).

l :- 
    Episode_1 = [
    diff_1(0, 1)
    ]/[],

    Episode_2 = [
    diff_2(0, 2)
    ]/[],

    Episode_3 = [
    diff_3(0, 3)
    ]/[],

    Episode_4 = [
    diff_4(0, 4)
    ]/[],

    Episode_5 = [
    diff_5(0, 5)
    ]/[],

    Episode_6 = [
    diff_6(0, 6)
    ]/[],

    Episode_7 = [
    diff_7(0, 7)
    ]/[],

    Episode_8 = [
    diff_8(0, 8)
    ]/[],

    Episode_9 = [
    diff_9(0, 9)
    ]/[],

    learn_seq([Episode_4, Episode_9, Episode_1, Episode_7,Episode_2, Episode_3, Episode_5, Episode_8, Episode_6], Prog),
    pprint(Prog).


m :-  
    Episode_1 = [
    diff_1(0, 1)
    ]/[],

    Episode_2 = [
    diff_2(0, 2)
    ]/[],

    Episode_3 = [
    diff_3(0, 3)
    ]/[],

    Episode_4 = [
    diff_4(0, 4)
    ]/[],

    Episode_5 = [
    diff_5(0, 5)
    ]/[],

    Episode_6 = [
    diff_6(0, 6)
    ]/[],

    Episode_7 = [
    diff_7(0, 7)
    ]/[],

    Episode_8 = [
    diff_8(0, 8)
    ]/[],

    Episode_9 = [
    diff_9(0, 9)
    ]/[],

    learn_seq([Episode_9, Episode_3, Episode_8, Episode_7, Episode_6, Episode_2, Episode_5, Episode_4, Episode_1], Prog),
    pprint(Prog).

n :- 
    Episode_1 = [
    diff_1(0, 1)
    ]/[],

    Episode_2 = [
    diff_2(0, 2)
    ]/[],

    Episode_3 = [
    diff_3(0, 3)
    ]/[],

    Episode_4 = [
    diff_4(0, 4)
    ]/[],

    Episode_5 = [
    diff_5(0, 5)
    ]/[],

    Episode_6 = [
    diff_6(0, 6)
    ]/[],

    Episode_7 = [
    diff_7(0, 7)
    ]/[],

    Episode_8 = [
    diff_8(0, 8)
    ]/[],

    Episode_9 = [
    diff_9(0, 9)
    ]/[],

    learn_seq([Episode_9, Episode_1, Episode_3, Episode_8, Episode_7, Episode_6, Episode_5, Episode_4, Episode_2], Prog),
    pprint(Prog).

t :- 
     a.  
    % b.
    % c.
    % d. 
    % e.  
    % f.  
    % g.  
    % h.  
    % i.
    % random ordered
    % j. 
    % k. 
    % l. 
    % m. 
    % n.

:- time(t).

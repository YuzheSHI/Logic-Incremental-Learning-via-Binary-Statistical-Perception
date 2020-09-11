:-use_module('metagol').

metagol: functional.

func_test(Atom1, Atom2, Condition):-
    Atom1 = [P, X, Y], 
    Atom2 = [P, X, Z],
    Condition = (Y = Z).

metarule(
    [P, Q],
    [P, X, Y],
    [[Q, X, Y]]
).

metarule(
    [P, Q, R],
    [P, X, Y],
    [[Q, X, Z], [R, Z, Y]]  
).

body_pred(succ/2).


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

    learn_seq([
        Episode_2 
        %Episode_3 
        %Episode_4 
        %Episode_5
        %Episode_6
        %Episode_7
        %Episode_8
        %Episode_9
        %Episode_1
        ], Prog),
    pprint(Prog).

:- time(b).
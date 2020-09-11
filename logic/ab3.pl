:-use_module('metagol').

% use background knowledge
body_pred(less/2).
body_pred(step/2).

% use metarules
metarule(
    ident, 
    [P, Q], 
    [P, A, B], 
    [[Q, A, B]]
).
metarule(
    chain, 
    [P, Q, R], 
    [P, A, B], 
    [[Q, A, C],[R, C, B]]
).
metarule(
    chain_3,
    [P, Q, R],
    [P, A, B, C],
    [[Q, A, B, T],[R, A, B, C]]
).

% background knowledge

zero(X).
one(X).
two(X).
three(X).
four(X).
five(X).
six(X).
seven(X).
eight(X).
nine(X).



less(zero(X), one(X)).
less(one(X), two(X)).
less(two(X), three(X)).
less(three(X), four(X)).
less(four(X), five(X)).
less(five(X), six(X)).
less(six(X), seven(X)).
less(seven(X), eight(X)).
less(eight(X), nine(X)).

abs(X, R) :-
    (X < 0) -> R is (-X).

step(A, B) :-
    less(A, B)



a :- 
    Episode_1 = [
        less_than(zero(X), one(X)),
        less_than(zero(X), nine(X)),
        less_than(one(X), eight(X)),
        less_than(one(X), three(X)),
        less_than(two(X), five(X)),
        less_than(three(X), eight(X)),
        less_than(four(X), seven(X)),
        less_than(five(X), eight(X)),
        less_than(five(X), nine(X)),
        less_than(seven(X), eight(X)),
        less_than(seven(X), nine(X)),
        less_than(eight(X), nine(X))
    ]/[],

    Episode_2 = [
        diff(zero(X), one(X), 1)
        % diff(zero(X), two(X), 2),
        % diff(zero(X), three(X), 3),
        % diff(zero(X), four(X), 4),
        % diff(zero(X), five(X), 5),
        % diff(zero(X), six(X), 6),
        % diff(zero(X), seven(X), 7),
        % diff(zero(X), eight(X), 8),
        % diff(zero(X), nine(X), 9)
    ]/[],

    learn_seq([Episode_1, Episode_2], Prog),
    pprint(Prog).

:- time(a).

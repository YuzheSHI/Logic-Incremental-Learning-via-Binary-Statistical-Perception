from pyswip import Prolog, Functor, Variable, Query


train_data = [
    "a"
    # "b", 
    # "c", 
    # "d", 
    # "e", 
    # "f", 
    # "g", 
    # "h", 
    # "i"
]


for i in train_data: 
    p = Prolog()
    p.consult("ab5")
    print(list(p.query(i)))
    



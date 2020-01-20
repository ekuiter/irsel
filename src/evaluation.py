from irsel import *
from helpers import *

# Irsel<wxyz>: w = binary/TF, x = 0/200 dimensions, y = 1/2 iterations, z = score 1e-1/1e-8
smart_0 = "nfc"
smart_1 = "bfc"
dim_0 = 0
dim_1 = 200
it_0 = 1
it_1 = 2
score_0 = 1e-1
score_1 = 1e-8
max_0 = 100
max_1 = 10

# normalized TF-IDF, no LSI
class Irsel0000(Irsel):
    __init__ = lambda self: super().__init__(smart=smart_0, dimensions=dim_0, iterations=it_0, select_until=All(Any(ScoreBelow(score_0), PremiseNumberAtLeast(max_0))))
    __str__ = lambda self: "irsel0000"

class Irsel0001(Irsel):
    __init__ = lambda self: super().__init__(smart=smart_0, dimensions=dim_0, iterations=it_0, select_until=All(Any(ScoreBelow(score_1), PremiseNumberAtLeast(max_0))))
    __str__ = lambda self: "irsel0001"

class Irsel0010(Irsel):
    __init__ = lambda self: super().__init__(smart=smart_0, dimensions=dim_0, iterations=it_1, select_until=All(Any(ScoreBelow(score_0), PremiseNumberAtLeast(max_1))))
    __str__ = lambda self: "irsel0010"

class Irsel0011(Irsel):
    __init__ = lambda self: super().__init__(smart=smart_0, dimensions=dim_0, iterations=it_1, select_until=All(Any(ScoreBelow(score_1), PremiseNumberAtLeast(max_1))))
    __str__ = lambda self: "irsel0011"

# normalized TF-IDF, 200-dimensional LSI
class Irsel0100(Irsel):
    __init__ = lambda self: super().__init__(smart=smart_0, dimensions=dim_1, iterations=it_0, select_until=All(Any(ScoreBelow(score_0), PremiseNumberAtLeast(max_0))))
    __str__ = lambda self: "irsel0100"

class Irsel0101(Irsel):
    __init__ = lambda self: super().__init__(smart=smart_0, dimensions=dim_1, iterations=it_0, select_until=All(Any(ScoreBelow(score_1), PremiseNumberAtLeast(max_0))))
    __str__ = lambda self: "irsel0101"

class Irsel0110(Irsel):
    __init__ = lambda self: super().__init__(smart=smart_0, dimensions=dim_1, iterations=it_1, select_until=All(Any(ScoreBelow(score_0), PremiseNumberAtLeast(max_1))))
    __str__ = lambda self: "irsel0110"

class Irsel0111(Irsel):
    __init__ = lambda self: super().__init__(smart=smart_0, dimensions=dim_1, iterations=it_1, select_until=All(Any(ScoreBelow(score_1), PremiseNumberAtLeast(max_1))))
    __str__ = lambda self: "irsel0111"

# normalized binary IDF, no LSI
class Irsel1000(Irsel):
    __init__ = lambda self: super().__init__(smart=smart_1, dimensions=dim_0, iterations=it_0, select_until=All(Any(ScoreBelow(score_0), PremiseNumberAtLeast(max_0))))
    __str__ = lambda self: "irsel1000"

class Irsel1001(Irsel):
    __init__ = lambda self: super().__init__(smart=smart_1, dimensions=dim_0, iterations=it_0, select_until=All(Any(ScoreBelow(score_1), PremiseNumberAtLeast(max_0))))
    __str__ = lambda self: "irsel1001"

class Irsel1010(Irsel):
    __init__ = lambda self: super().__init__(smart=smart_1, dimensions=dim_0, iterations=it_1, select_until=All(Any(ScoreBelow(score_0), PremiseNumberAtLeast(max_1))))
    __str__ = lambda self: "irsel1010"

class Irsel1011(Irsel):
    __init__ = lambda self: super().__init__(smart=smart_1, dimensions=dim_0, iterations=it_1, select_until=All(Any(ScoreBelow(score_1), PremiseNumberAtLeast(max_1))))
    __str__ = lambda self: "irsel1011"

# normalized binary IDF, 200-dimensional LSI
class Irsel1100(Irsel):
    __init__ = lambda self: super().__init__(smart=smart_1, dimensions=dim_1, iterations=it_0, select_until=All(Any(ScoreBelow(score_0), PremiseNumberAtLeast(max_0))))
    __str__ = lambda self: "irsel1100"

class Irsel1101(Irsel):
    __init__ = lambda self: super().__init__(smart=smart_1, dimensions=dim_1, iterations=it_0, select_until=All(Any(ScoreBelow(score_1), PremiseNumberAtLeast(max_0))))
    __str__ = lambda self: "irsel1101"

class Irsel1110(Irsel):
    __init__ = lambda self: super().__init__(smart=smart_1, dimensions=dim_1, iterations=it_1, select_until=All(Any(ScoreBelow(score_0), PremiseNumberAtLeast(max_1))))
    __str__ = lambda self: "irsel1110"

class Irsel1111(Irsel):
    __init__ = lambda self: super().__init__(smart=smart_1, dimensions=dim_1, iterations=it_1, select_until=All(Any(ScoreBelow(score_1), PremiseNumberAtLeast(max_1))))
    __str__ = lambda self: "irsel1111"
#!/usr/bin/python3

import sys
from gavel.dialects.tptp.parser import TPTPProblemParser
from gavel.prover.eprover.interface import EProverInterface
from gavel.selection.selector import Selector

class Irsel(Selector):
    def select(self, problem):
        return problem

prover = EProverInterface()
parser = TPTPProblemParser()
selector = Irsel()

if len(sys.argv) < 2:
    sys.exit('no problem file given')
f = sys.argv[1]
with open(f) as file:
    lines = file.readlines()
problem = next(parser.parse(lines), None)
problem = selector.select(problem)
proof = prover.prove(problem)
for s in proof.steps:
    print("{name}: {formula}".format(name=s.name, formula=s.formula))
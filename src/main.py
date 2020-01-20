#!/usr/bin/python3

from typing import Tuple, Iterable
from logging import basicConfig, INFO
from argparse import ArgumentParser
from subprocess import CalledProcessError, TimeoutExpired
from itertools import combinations

from gavel.dialects.base.parser import ProblemParser
from gavel.dialects.tptp.parser import TPTPProblemParser
from gavel.prover.base.interface import BaseProverInterface
from gavel.prover.eprover.interface import EProverInterface
from gavel.selection.selector import Selector, Sine
from gavel.logic.problem import Problem, Sentence, AnnotatedFormula
from gavel.logic.proof import Proof

from helpers import *
from irsel import *
from evaluation import *

class Main:
    def parse(self, parser: ProblemParser, filename: str) -> Problem:
        """Parses a problem from a file."""
        with Message("Parsing problem"):
            problem = next(parser.parse(open(filename)), None)
        if problem is None:
            raise ValueError("could not parse problem")
        return problem

    def prove(self, problem: Problem, selector: Selector, prover: BaseProverInterface) -> Tuple[Proof, Timer, Timer, int, int, Iterable[Sentence]]:
        """Selects relevant axioms and attempts to construct a proof."""
        printq("Performing axiom selection.")
        with Timer() as selection_timer:
            reduced_problem = selector.select(problem)
        printq(f"Selected {len(reduced_problem.premises)} of {len(problem.premises)} axioms.")
        if get_verbose() and isinstance(selector, Sine):
            for premise in reduced_problem.premises:
                print(f"{premise.name}: {truncate(str(premise.formula), n=verbose_axiom_length)}")

        with Message("Attempting proof") as message:
            try:
                proof = prover.prove(reduced_problem)
            except CalledProcessError:
                proof = None
            except TimeoutExpired:
                printq("Timeout while attempting proof.")
                proof = None
            proof_timer = message.timer
            premise_num = len(problem.premises)
            reduced_premise_num = len(reduced_problem.premises)
            return proof, selection_timer, proof_timer, premise_num, reduced_premise_num, reduced_problem.premises

    def __init__(self):
        arg_parser = ArgumentParser("irsel")
        arg_parser.add_argument("problem_file", help="TPTP problem file", nargs='+')
        arg_parser.add_argument("-s", "--selector", action="append", help="identity, sine or irsel (default)")
        arg_parser.add_argument("-a", "--all", action="store_true", help="select with identity, sine and irsel")
        arg_parser.add_argument("-e", "--evaluate", action="store_true", help="select with several irsel variants")
        arg_parser.add_argument("-c", "--compare", action="store_true", help="compare pairs of selectors")
        arg_parser.add_argument("-t", "--timeout", action="store", type=float, help="EProver timeout in seconds (default: none)")
        arg_parser.add_argument("-v", "--verbose", action="store_true", help="print verbose information")
        arg_parser.add_argument("-q", "--quiet", action="store_true", help="print less information")
        # the following arguments are only active when using the irsel selector directly
        arg_parser.add_argument("-d", "--dimensions", action="store", type=int, help="number of latent dimensions, 0 to disable LSI", default=Irsel.default_dimensions)
        arg_parser.add_argument("-n", "--iterations", action="store", type=int, help="number of querying iterations", default=Irsel.default_iterations)
        arg_parser.add_argument("--score", action="store", type=float, help="minimum score to select axiom (per iteration)", default=Irsel.default_score)
        arg_parser.add_argument("--max", action="store", type=int, help="maximum number of selected axioms (per iteration)", default=Irsel.default_max)
        arg_parser.add_argument("--smart", action="store", help="SMART Information Retrieval System mnemonic", default=Irsel.default_smart)
        arg_parser.add_argument("-i", "--inspect", action="append", help="name of axiom to inspect")
        args = arg_parser.parse_args()
        if args.verbose and not args.quiet:
            set_verbose(True)
            basicConfig(format="%(levelname)s: %(message)s", level=INFO)
        if args.quiet:
            set_quiet(True)

        parser = TPTPProblemParser()
        prover = EProverInterface(timeout=args.timeout)
        irsel_kwargs = {"dimensions": args.dimensions, "iterations": args.iterations, "inspect_premises": args.inspect,
            "select_until": Irsel.default_select_until(args.score, args.max), "smart": args.smart}

        selector_map = {"identity": Selector, "sine": Sine, "irsel": Irsel}
        args.selector = ["identity", "sine", "irsel"] if args.all else args.selector
        selectors = [(selector_map[selector_name](**irsel_kwargs) if selector_name == "irsel" else selector_map[selector_name]())
            for selector_name in (args.selector if args.selector else ([] if args.evaluate else ["irsel"])) if selector_name in selector_map]
        if args.evaluate:
            selectors.extend([
                Irsel0000(), Irsel0001(), Irsel0010(), Irsel0011(), Irsel0100(), Irsel0101(), Irsel0110(), Irsel0111(),
                Irsel1000(), Irsel1001(), Irsel1010(), Irsel1011(), Irsel1100(), Irsel1101(), Irsel1110(), Irsel1111()])
        results = {}

        for problem_file in args.problem_file:
            print()
            print(f"- Problem {problem_file} -")
            problem = self.parse(parser, problem_file)

            for selector in selectors:
                printq()
                selector_name = get_selector_name(selector)
                printq(f"- {selector_name} selector -")
                proof, selection_timer, proof_timer, premise_num, reduced_premise_num, reduced_premises = self.prove(
                    problem=problem, selector=selector, prover=prover)
                results[selector_name] = (proof, selection_timer, proof_timer, premise_num, reduced_premise_num, reduced_premises)
                if proof:
                    printq(f"Proof found with {len(proof.steps)} steps.")
                else:
                    printq(f"No proof found for conjecture. Maybe the {selector_name} selector is too strict?")

            if args.compare:
                map_name = lambda f: str(f.name)
                jaccard = lambda p1, p2: len(p1.intersection(p2)) / len(p1.union(p2))
                print()
                print("- comparison -")
                print("{0:15} {1:15} {2:10} {3:10} {4:10} {5}".format("selector 1", "selector 2", "Jaccard", "both", "only 1", "only 2"))
                for s1, s2 in sorted(combinations(selectors, 2),
                    key=lambda e: -jaccard(set(results[get_selector_name(e[0])][5]), set(results[get_selector_name(e[1])][5]))):
                    if type(s1) != Selector and type(s2) != Selector and not (isinstance(s1, Irsel) and isinstance(s2, Irsel)):
                        s1_name = get_selector_name(s1)
                        s2_name = get_selector_name(s2)
                        _, _, _, _, _, p1 = results[s1_name]
                        _, _, _, _, _, p2 = results[s2_name]
                        p1 = set(p1)
                        p2 = set(p2)
                        print("{0:15} {1:15} {2:10} {3:10} {4:10} {5}".format(s1_name, s2_name,
                            f"{round(jaccard(p1, p2) * 100, 2)}%",
                            str(len(list(map(map_name, p1.intersection(p2))))),
                            str(len(list(map(map_name, p1.difference(p2))))),
                            str(len(list(map(map_name, p2.difference(p1)))))))

            print()
            print("- results -")
            print("{0:15} {1:15} {2:15} {3:15} {4}".format("selector", "proof steps", "selection time", "proof time", "selection ratio"))
            for selector_name in results:
                proof, selection_timer, proof_timer, premise_num, reduced_premise_num, reduced_premises = results[selector_name]
                print("{0:15} {1:15} {2:15} {3:15} {4}% ({5} of {6})".format(
                    selector_name, str(len(proof.steps)) if proof else "-", str(selection_timer), str(proof_timer),
                    round(reduced_premise_num / premise_num * 100, 2), reduced_premise_num, premise_num))

if __name__ == "__main__":
    AnnotatedFormula.__hash__ = lambda self: id(self) # hash AnnotatedFormulas by reference (to enable usage in sets)
    Main()
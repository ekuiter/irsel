#!/usr/bin/python3

from typing import List, Tuple, Iterable, Callable
from logging import basicConfig, INFO
from argparse import ArgumentParser
from subprocess import CalledProcessError, TimeoutExpired
from timeit import default_timer
from datetime import timedelta
from itertools import tee, takewhile, combinations
from functools import partial, reduce

from gavel.dialects.base.parser import ProblemParser
from gavel.dialects.tptp.parser import TPTPProblemParser
from gavel.prover.base.interface import BaseProverInterface
from gavel.prover.eprover.interface import EProverInterface
from gavel.selection.selector import Selector, Sine
from gavel.logic.problem import Problem, Sentence, AnnotatedFormula
from gavel.logic.proof import Proof

from gensim.corpora import Dictionary, MmCorpus
from gensim.models import TfidfModel, LsiModel
from gensim.similarities import Similarity
from gensim.test.utils import get_tmpfile
from gensim.matutils import corpus2dense

# global variables
verbose = False
quiet = False
verbose_axiom_number = 20
verbose_axiom_length = 80

# readable type hints
TokenList = List[str] # e.g., the token list for "p(a,b)" is ["p", "a", "b"]
Vector    = List[Tuple[int, int]] # e.g., the vector for "p(a,b)" with dictionary p=0, a=1, b=2 is [(0, 1), (1, 1), (2, 1)]
Corpus    = Iterable[Vector] # a lazy term-document matrix

# helpers
identity = lambda x: x
first    = lambda e: e[0]
second   = lambda e: e[1]
truncate = lambda s, n: (s[:n] + "...") if len(s) > n else s
printq   = lambda *args, **kwargs: print(*args, **kwargs) if not quiet else None

# hash AnnotatedFormulas by reference (to enable usage in sets)
AnnotatedFormula.__hash__ = lambda self: id(self)

def do_to(value, *functions):
    """Threads a value through a chain of functions, inspired by https://clojure.org/guides/threading_macros."""
    return reduce(lambda f, g: lambda x: g(f(x)), functions, identity)(value)

def get_selector_name(selector):
    """Returns a human-readable selector name."""
    if type(selector) == Selector:
        return "identity"
    elif type(selector).__str__ is not object.__str__:
        return str(selector)
    else:
        return type(selector).__name__.lower()

class Timer:
    """Measures elapsed time."""

    def __init__(self):
        self.start_time = default_timer()

    def __call__(self):
        return self.timer() - self.start_time

    def __str__(self):
        return str(timedelta(seconds=self()))

    def __enter__(self):
        self.start_time = default_timer()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    def timer(self):
        return self.end_time if hasattr(self, "end_time") else default_timer()

    def stop(self):
        self.end_time = self.timer()

class Message:
    """Shows progress for long-running operations."""

    def __init__(self, message, show_done=True):
        self.message = message
        self.show_done = show_done

    def __enter__(self):
        self.timer = Timer()
        printq(f"{self.message} ... ", end="" if self.show_done else "\n", flush=True)
        return self

    def __exit__(self, type, value, traceback):
        self.timer.stop()
        if self.show_done:
            printq(f"done ({self.timer} seconds)." if self.timer() >= 1 else "done.")

class PremiseNumberAtLeast:
    """Triggers after a fixed number of premises."""

    def __init__(self, number):
        self.number = number
    
    def __call__(self, index, total, score, score_sum):
        return index >= self.number

class ScoreBelow:
    """Triggers when the premises fall below a given score."""

    def __init__(self, score):
        self.score = score
    
    def __call__(self, index, total, score, score_sum):
        return score < self.score

class PremisePercentageAtLeast:
    """Triggers when a given percentage of all premises has been selected."""

    def __init__(self, percentage):
        self.percentage = percentage
        self.premises = 0
    
    def __call__(self, index, total, score, score_sum):
        self.premises += 1
        return self.premises >= total * self.percentage

class ScorePercentageAtLeast:
    """Triggers when a given percentage of the cumulated similarity scores has been selected."""

    def __init__(self, percentage):
        self.percentage = percentage
        self.score = 0
    
    def __call__(self, index, total, score, score_sum):
        self.score += score
        return self.score >= score_sum * self.percentage

class All:
    """Triggers when all of the given functions are triggered."""

    def __init__(self, *args):
        self.functions = args

    def __call__(self, index, total, score, score_sum):
        return reduce(lambda acc, fn: acc and fn(index, total, score, score_sum), self.functions, True)

class Any:
    """Triggers when any of the given functions is triggered."""

    def __init__(self, *args):
        self.functions = args

    def __call__(self, index, total, score, score_sum):
        return reduce(lambda acc, fn: acc or fn(index, total, score, score_sum), self.functions, False)

class Irsel(Selector):
    """Selects relevant axioms with information retrieval techniques."""

    # defaults
    default_dimensions = 200
    default_iterations = 2
    default_score = 1e-8
    default_max = 10
    default_select_until = lambda score, _max: All(Any(ScoreBelow(score), PremiseNumberAtLeast(_max)))
    default_smart = "nfc" # normalized TF-IDF
    index_cache = None

    def __init__(self, dimensions=default_dimensions, iterations=default_iterations,
        select_until=default_select_until(default_score, default_max), smart=default_smart, inspect_premises=[]):
        self.dimensions = dimensions
        self.iterations = iterations
        self.select_until = select_until
        self.inspect_premises = inspect_premises
        self.smart = smart

    def __str__(self):
        return "irsel"

    def tokenize_formula(self, formula: Sentence) -> TokenList:
        """Extracts all symbols (predicates, functors, and constants) from a given formula."""
        return list(map(str, formula.symbols()))

    def tokenize_premises(self, premises: Iterable[Sentence]) -> Iterable[TokenList]:
        """Extracts all symbols from all premises of a given problem."""
        # We use generators instead of lists to optimize performance, see
        # https://radimrehurek.com/gensim/auto_examples/core/run_corpora_and_vector_spaces.html#corpus-streaming-tutorial.
        return (self.tokenize_formula(premise) for premise in premises)

    def persist_corpus(self, corpus: Corpus, key: str="corpus") -> Corpus:
        """Takes a transient corpus generator and persists it on disk. Only necessary when using a corpus more than once."""
        with Message(f"Storing {key} corpus"):
            f = get_tmpfile(f"irsel_{key}")
            # By serializing a corpus to disk, we can read it multiple times (which is impossible with a generator)
            # without having to load it into RAM as a whole at any time.
            MmCorpus.serialize(f, corpus)
            corpus = MmCorpus(f) # this instance can be consumed as often as we want
        printq(corpus)
        return corpus

    def build_corpus(self, premises: Iterable[Sentence]) -> Tuple[Dictionary, Corpus]:
        """Builds a TF corpus from given premises."""

        with Message("Building dictionary of symbols"):
            # Establish a mapping from symbols to unique IDs. All following data structures will make use of this mapping.
            dictionary = Dictionary(self.tokenize_premises(premises))
        printq(dictionary)

        with Message("Building corpus of premises"):
            # Transform each document (premise) to a sparse bag-of-words vector (containing term frequencies).
            # In other words, this builds a sparse term-document matrix (additionally, it is lazy due to generators).
            corpus = (dictionary.doc2bow(tokenized_premise) for tokenized_premise in self.tokenize_premises(premises))

        corpus = self.persist_corpus(corpus) # make the corpus instance reusable
        return dictionary, corpus

    def transform_corpus(self, dictionary: Dictionary, corpus: Corpus) -> Tuple[Corpus, Callable[[TokenList], Vector]]:
        """Transforms a TF corpus to TF-IDF and then to LSI vector space."""

        with Message("Initializing TF-IDF model"):
            # Transforms term-document matrix with integer entries tf(i,j) (term i in document j) to a TF-IDF matrix
            # with float entries tf-idf(i,j) = tf(i,j) * log2(D/df(i)), where D is the total number of documents and
            # df(i) the number of documents containing the term i. The new document vectors are normalized.
            # Usually we should pass the corpus here. However, the dictionary already contains all required information.
            tfidf_model = TfidfModel(None, id2word=dictionary, dictionary=dictionary, smartirs=self.smart)
        printq(tfidf_model)

        with Message("Applying TF-IDF model"):
            # Perform actual transformation from TF to TF-IDF vector space ([] is overloaded to mean "apply transformation").
            tfidf_corpus = tfidf_model[corpus]

        if self.dimensions:
            with Message("Initializing LSI model"):
                # Apply latent semantic indexing (that is, a singular value decomposition on the TF-IDF matrix) to discover
                # "topics" or clusters of co-occuring symbols. Reduces dimensions to num_topics with low-rank approximation.
                lsi_model = LsiModel(tfidf_corpus, id2word=dictionary, num_topics=self.dimensions)
            printq(lsi_model)

            with Message("Applying LSI model"):
                # Transform X = tfidf_corpus from TF-IDF to LSI space. For X = U*S*V^T, this computes U^-1*X = V*S.
                # This reduces dimensions and "squishes" similar "topics"/related symbols into the same dimension.
                lsi_corpus = lsi_model[tfidf_corpus]

            if verbose:
                U = lsi_model.projection.u # relates terms (rows) to topics (columns)
                S = lsi_model.projection.s # ranks relevance of topics
                V = corpus2dense(lsi_corpus, len(S)).T / S # relates documents (rows) to topics (vectors)
                print(f"U = {U.shape}, S = {S.shape}, V = {V.shape}")

            # to query the index later, we have to transform the queried formula the same way we transformed the corpus
            new_corpus = lsi_corpus
            query_transformer = lambda tokenized_formula: lsi_model[tfidf_model[dictionary.doc2bow(tokenized_formula)]]
        else:
            printq("Skipping LSI model.")
            new_corpus = tfidf_corpus
            query_transformer = lambda tokenized_formula: tfidf_model[dictionary.doc2bow(tokenized_formula)]

        return new_corpus, query_transformer

    def build_index(self, premises: Iterable[Sentence]) -> Tuple[Similarity, Callable[[TokenList], Vector], Iterable[Sentence]]:
        """Builds an index from given premises that can be used to answer similarity queries."""

        if Irsel.index_cache:
            # if an index has already been built for these TF-IDF parameters, reuse it
            cached_smart, cached_dimensions, cached_index, cached_query_transformer, cached_premises = Irsel.index_cache
            if cached_smart == self.smart and cached_dimensions == self.dimensions and cached_premises is premises:
                printq("Hitting index cache.")
                return cached_index, cached_query_transformer, cached_premises
            else:
                printq("Skipping index cache.")

        dictionary, corpus = self.build_corpus(premises) # create a term-document matrix
        corpus, query_transformer = self.transform_corpus(dictionary, corpus) # apply TF-IDF and LSI models

        with Message("Storing index"):
            # Builds an index which we can compare queries against.
            index = Similarity(get_tmpfile(f"irsel_index"), corpus, num_features=len(dictionary))
        printq(index)

        # allows us to reuse this index for later proof attempts with the same parameters
        Irsel.index_cache = self.smart, self.dimensions, index, query_transformer, premises
        return index, query_transformer, premises

    def query_index(self, index: Similarity, query_transformer: Callable[[TokenList], Vector],
        premises: Iterable[Sentence], query: Sentence) -> Iterable[Sentence]:
        """Queries an index for the premises that match a given formula best."""

        with Message(f"Querying index for formula {query.name}"):
            # extract symbols from the formula, then transform the token list to an LSI vector
            # and query the index for the formula's similarities to all premises
            similarities = index[query_transformer(self.tokenize_formula(query))]
            score_sum = reduce(lambda x, y: x + y, similarities, 0)

        # show premise scores of interest
        if self.inspect_premises:
            print("Inspected axiom scores:")
            for idx, score in enumerate(similarities):
                premise_name = premises[idx].name
                if premise_name in self.inspect_premises:
                    print(f"{score}\t{premise_name}")

        with Message("Selecting best-matching axioms", show_done=not verbose):
            return do_to(
                similarities, # take the similarity scores, then
                enumerate, # pair scores with premise indices
                partial(sorted, key=lambda e: -e[1]), # sort by descending similarity score
                enumerate, # pair with counter so we can terminate early
                partial(takewhile, lambda e: not self.select_until(index=e[0], # select premises until done
                    total=len(premises), score=e[1][1], score_sum=score_sum)),
                partial(map, lambda e: (e[0], (premises[e[1][0]], e[1][1]))), # map premise index to premise
                partial(map, lambda e: (print(f"{e[1][1]}\t{e[1][0].name}: {truncate(str(e[1][0].formula), n=verbose_axiom_length)}")
                    if e[0] < verbose_axiom_number else None, e)[1])
                    if verbose else identity, # print summary
                partial(map, second), # discard counter
                partial(map, first), # discard scores
                list # consume generator
            )

    def select(self, problem: Problem) -> Problem:
        """For a given problem with premises Ax and conjecture C, returns a reduced problem with
        premises Ax' and conjecture C such that Ax' is a subset of Ax.
        Ideally, Ax' should be a minimal subset of Ax with the property: C follows from Ax <=> C follows from Ax'.
        When using information retrieval techniques in the context of axiom selection, we may identify:
        - tokens = terms = symbols (that is, predicates, functors, and constants)
        - documents = formulas     (that is, the premises and the conjecture)
        - corpus = premises        (that is, the corpus or term-document matrix represents the ontology)"""

        # indexing phase: this (in principle) just has to be done once per ontology
        index, query_transformer, premises = self.build_index(problem.premises)

        # querying phase: only here do we depend on the conjecture
        reduced_premises = set()
        step = [problem.conjecture]
        for i in range(0, self.iterations):
            if self.iterations > 1:
                printq(f"Iteration {i + 1} of {self.iterations}:")
            step = set([new_formula for formula in step for new_formula in self.query_index(index, query_transformer, premises, query=formula)])
            step = step.difference(reduced_premises)
            if not step:
                printq("Reached fixed point.")
                break
            reduced_premises.update(step)

        return Problem(premises=reduced_premises, conjecture=problem.conjecture)

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
        if verbose and isinstance(selector, Sine):
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
            global verbose
            verbose = True
            basicConfig(format="%(levelname)s: %(message)s", level=INFO)
        if args.quiet:
            global quiet
            quiet = True

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
    Main()
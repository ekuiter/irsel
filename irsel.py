#!/usr/bin/python3

from typing import List, Tuple, Iterable, Callable
from logging import basicConfig, INFO
from argparse import ArgumentParser
from subprocess import CalledProcessError
from timeit import default_timer
from datetime import timedelta
from itertools import tee, takewhile
from functools import partial, reduce

from gavel.dialects.base.parser import ProblemParser
from gavel.dialects.tptp.parser import TPTPProblemParser
from gavel.prover.base.interface import BaseProverInterface
from gavel.prover.eprover.interface import EProverInterface
from gavel.selection.selector import Selector, Sine
from gavel.logic.problem import Problem, Sentence
from gavel.logic.proof import Proof

from gensim.corpora import Dictionary, MmCorpus
from gensim.models import TfidfModel, LsiModel
from gensim.similarities import Similarity
from gensim.test.utils import get_tmpfile
from gensim.matutils import corpus2dense

# global variables
verbose = False

# readable type hints
TokenList = List[str] # e.g., the token list for "p(a,b)" is ["p", "a", "b"]
Vector    = List[Tuple[int, int]] # e.g., the vector for "p(a,b)" with dictionary p=0, a=1, b=2 is [(0, 1), (1, 1), (2, 1)]
Corpus    = Iterable[Vector] # a lazy term-document matrix

# helpers
identity = lambda x: x
first    = lambda e: e[0]
second   = lambda e: e[1]

def do_to(value, *functions):
    """Threads a value through a chain of functions, inspired by https://clojure.org/guides/threading_macros."""
    return reduce(lambda f, g: lambda x: g(f(x)), functions, identity)(value)

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
        print(f"{self.message} ... ", end="" if self.show_done else "\n", flush=True)
        return self

    def __exit__(self, type, value, traceback):
        self.timer.stop()
        if self.show_done:
            print(f"done ({self.timer} seconds)." if self.timer() >= 1 else "done.")

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

    def __init__(self, dimensions=200, iterations=1, select_until=All(PremiseNumberAtLeast(10), Any(ScoreBelow(1e-10), PremiseNumberAtLeast(1000)))):
        self.dimensions = dimensions
        self.iterations = iterations
        self.select_until = select_until

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
        print(corpus)
        return corpus

    def build_corpus(self, premises: Iterable[Sentence]) -> Tuple[Dictionary, Corpus]:
        """Builds a TF corpus from given premises."""

        with Message("Building dictionary of symbols"):
            # Establish a mapping from symbols to unique IDs. All following data structures will make use of this mapping.
            dictionary = Dictionary(self.tokenize_premises(premises))
        print(dictionary)

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
            tfidf_model = TfidfModel(None, id2word=dictionary, dictionary=dictionary)
        print(tfidf_model)

        with Message("Applying TF-IDF model"):
            # Perform actual transformation from TF to TF-IDF vector space ([] is overloaded to mean "apply transformation").
            tfidf_corpus = tfidf_model[corpus]

        with Message("Initializing LSI model"):
            # Apply latent semantic indexing (that is, a singular value decomposition on the TF-IDF matrix) to discover
            # "topics" or clusters of co-occuring symbols. Reduces dimensions to num_topics with low-rank approximation.
            lsi_model = LsiModel(tfidf_corpus, id2word=dictionary, num_topics=self.dimensions)
        print(lsi_model)

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
        query_transformer = lambda tokenized_formula: lsi_model[tfidf_model[dictionary.doc2bow(tokenized_formula)]]
        return lsi_corpus, query_transformer

    def build_index(self, premises: Iterable[Sentence]) -> Tuple[Similarity, Callable[[TokenList], Vector], Iterable[Sentence]]:
        """Builds an index from given premises that can be used to answer similarity queries."""

        dictionary, corpus = self.build_corpus(premises) # create a term-document matrix
        corpus, query_transformer = self.transform_corpus(dictionary, corpus) # apply TF-IDF and LSI models

        with Message("Storing index"):
            # Builds an index which we can compare queries against.
            index = Similarity(get_tmpfile(f"irsel_index"), corpus, num_features=len(dictionary))
        print(index)

        return index, query_transformer, premises

    def query_index(self, index: Similarity, query_transformer: Callable[[TokenList], Vector],
        premises: Iterable[Sentence], query: Sentence) -> Iterable[Sentence]:
        """Queries an index for the premises that match a given formula best."""

        with Message("Querying index for formula"):
            # extract symbols from the formula, then transform the token list to an LSI vector
            # and query the index for the formula's similarities to all premises
            similarities = index[query_transformer(self.tokenize_formula(query))]
            score_sum = reduce(lambda x, y: x + y, similarities, 0)
        
        with Message("Selecting best-matching axioms", show_done=not verbose):
            return do_to(
                similarities, # take the similarity scores, then
                enumerate, # pair scores with premise indices
                partial(sorted, key=lambda e: -e[1]), # sort by descending similarity score
                enumerate, # pair with counter so we can terminate early
                partial(takewhile, lambda e: not self.select_until(index=e[0], # select premises until done
                    total=len(premises), score=e[1][1], score_sum=score_sum)),
                partial(map, second), # discard counter
                partial(map, lambda e: (premises[e[0]], e[1])), # map premise index to premise
                partial(map, lambda e: (print(f"{e[1]}\t{e[0].formula}"), e)[1]) if verbose else identity, # print summary
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
        reduced_premises = [] # TODO: use set() to eliminate duplicates, but AnnotatedFormulas are not hashable
        step = [problem.conjecture] # TODO: iteration does not work yet as intended
        for i in range(0, self.iterations):
            if self.iterations > 1:
                print(f"Iteration {i + 1} of {self.iterations}:")
            step = (new_formula for formula in step for new_formula in self.query_index(index, query_transformer, premises, query=formula))
            len_before = len(reduced_premises)
            reduced_premises.extend(step)
            if len(reduced_premises) == len_before:
                print("Reached fixed point.")
                break

        return Problem(premises=reduced_premises, conjecture=problem.conjecture)

class Main:
    def parse(self, parser: ProblemParser, filename: str) -> Problem:
        """Parses a problem from a file."""
        with Message("Parsing problem"):
            problem = next(parser.parse(open(filename)), None)
        if problem is None:
            raise ValueError("could not parse problem")
        return problem

    def prove(self, problem: Problem, selector: Selector, prover: BaseProverInterface) -> Tuple[Proof, Timer, Timer, int, int]:
        """Selects relevant axioms and attempts to construct a proof."""
        print("Performing axiom selection.")
        with Timer() as selection_timer:
            reduced_problem = selector.select(problem)
        print(f"Selected {len(reduced_problem.premises)} of {len(problem.premises)} axioms.")

        with Message("Attempting proof") as message:
            try:
                proof = prover.prove(reduced_problem)
            except CalledProcessError:
                proof = None
            proof_timer = message.timer
            premise_num = len(problem.premises)
            reduced_premise_num = len(reduced_problem.premises)
            return proof, selection_timer, proof_timer, premise_num, reduced_premise_num

    def __init__(self):
        parser = ArgumentParser("irsel")
        parser.add_argument("problem_file", help="TPTP problem file")
        parser.add_argument("-s", "--selector", action="append", help="identity, sine or irsel (default)")
        parser.add_argument("-v", "--verbose", action="store_true", help="print verbose information")
        args = parser.parse_args()
        if args.verbose:
            global verbose
            verbose = True
            basicConfig(format="%(levelname)s: %(message)s", level=INFO)

        selector_map = {"identity": Selector, "sine": Sine, "irsel": Irsel}
        args.selector = ["identity", "sine", "irsel"] if args.selector == ["all"] else args.selector
        selectors = [selector_map[selector_name]() for selector_name in
            (args.selector if args.selector else ["irsel"]) if selector_name in selector_map]
        results = {}

        problem = self.parse(TPTPProblemParser(), args.problem_file)

        for selector in selectors:
            print()
            selector_name = type(selector).__name__.lower() if type(selector) != Selector else "identity"
            print(f"- {selector_name} selector -")
            proof, selection_timer, proof_timer, premise_num, reduced_premise_num = self.prove(
                problem=problem, selector=selector, prover=EProverInterface())
            results[selector_name] = (proof, selection_timer, proof_timer, premise_num, reduced_premise_num)
            if proof:
                print(f"Proof found with {len(proof.steps)} steps.")#
            else:
                print(f"No proof found for conjecture. Maybe the {selector_name} selector is too strict?")

        print()
        print("- summary -")
        print("{0:15} {1:15} {2:15} {3:15} {4}".format("selector", "proof steps", "selection time", "proof time", "selection ratio"))
        for selector_name in results:
            proof, selection_timer, proof_timer, premise_num, reduced_premise_num = results[selector_name]
            print("{0:15} {1:15} {2:15} {3:15} {4}% ({5} of {6})".format(
                selector_name, str(len(proof.steps)) if proof else "-", str(selection_timer), str(proof_timer),
                round(reduced_premise_num / premise_num * 100, 2), reduced_premise_num, premise_num))

if __name__ == "__main__":
    Main()
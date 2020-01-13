#!/usr/bin/python3

from typing import List, Tuple, Iterable
from logging import basicConfig, INFO
from argparse import ArgumentParser
from subprocess import CalledProcessError
from timeit import default_timer
from datetime import timedelta
from itertools import tee, takewhile

from gavel.dialects.base.parser import ProblemParser
from gavel.dialects.tptp.parser import TPTPProblemParser
from gavel.prover.base.interface import BaseProverInterface
from gavel.prover.eprover.interface import EProverInterface
from gavel.selection.selector import Selector, Sine
from gavel.logic.problem import Problem
from gavel.logic.logic import LogicElement
from gavel.logic.proof import Proof

from gensim.corpora import Dictionary, MmCorpus
from gensim.models import TfidfModel, LsiModel
from gensim.similarities import Similarity
from gensim.test.utils import get_tmpfile
from gensim.matutils import corpus2dense

VERBOSE = False

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

    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.timer = Timer()
        print(f"{self.message} ... ", end="", flush=True)
        return self

    def __exit__(self, type, value, traceback):
        self.timer.stop()
        print(f"done ({self.timer} seconds)." if self.timer() >= 1 else "done.")

class Irsel(Selector):
    """Selects relevant axioms with information retrieval techniques."""

    # readable type hints
    TokenList = List[str] # e.g., the token list for "p(a,b)" is ["p", "a", "b"]
    Vector = List[Tuple[int, int]] # e.g., the vector for "p(a,b)" with dictionary p=0, a=1, b=2 is [(0, 1), (1, 1), (2, 1)]
    Corpus = Iterable[Vector] # a lazy term-document matrix

    def tokenize_formula(self, formula: LogicElement) -> TokenList:
        """Extracts all symbols (predicates, functors, and constants) from a given formula."""
        return list(map(str, formula.symbols()))

    def tokenize_premises(self, problem: Problem) -> Iterable[TokenList]:
        """Extracts all symbols from all premises of a given problem."""
        # We use generators instead of lists to optimize performance, see
        # https://radimrehurek.com/gensim/auto_examples/core/run_corpora_and_vector_spaces.html#corpus-streaming-tutorial.
        return (self.tokenize_formula(premise) for premise in problem.premises)

    def persist_corpus(self, corpus: Corpus, key: str) -> Corpus:
        """Takes a transient corpus generator and persists it on disk. Only necessary when using a corpus more than once."""
        with Message(f"Storing {key} corpus"):
            f = get_tmpfile(f"irsel_{key}")
            # By serializing a corpus to disk, we can read it multiple times (which is impossible with a generator)
            # without having to load it into RAM as a whole at any time.
            MmCorpus.serialize(f, corpus)
            corpus = MmCorpus(f) # this instance can be consumed as often as we want
        print(corpus)
        return corpus

    def prepare_corpus(self, problem: Problem) -> Tuple[Dictionary, Corpus]:
        """Prepares a corpus from the premises of a given problem."""

        with Message("Building dictionary of symbols"):
            # Establish a mapping from symbols to unique IDs. All following data structures will make use of this mapping.
            dictionary = Dictionary(self.tokenize_premises(problem))
        print(dictionary)

        with Message("Building corpus of premises"):
            # Transform each document (premise) to a sparse bag-of-words vector (containing term frequencies).
            # In other words, this builds a sparse term-document matrix (additionally, it is lazy due to generators).
            corpus = (dictionary.doc2bow(premise) for premise in self.tokenize_premises(problem))

        corpus = self.persist_corpus(corpus, key="corpus") # make the corpus instance reusable
        return dictionary, corpus

    def select(self, problem: Problem) -> Problem:
        """For a given problem with premises Ax and conjecture C, returns a reduced problem with
        premises Ax' and conjecture C such that Ax' is a subset of Ax.
        Ideally, Ax' should be a minimal subset of Ax with the property: C follows from Ax <=> C follows from Ax'"""
        # When using information retrieval techniques in the context of axiom selection, we may identify:
        # - tokens = terms = symbols (that is, predicates, functors, and constants)
        # - documents = formulas     (that is, the premises and the conjecture)
        # - corpus = premises        (that is, the corpus or term-document matrix represents the ontology)

        dictionary, corpus = self.prepare_corpus(problem) # create a term-document matrix

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
            lsi_model = LsiModel(tfidf_corpus, id2word=dictionary, num_topics=200) # TODO: num_topics 200-500? (default 200)
        print(lsi_model)

        with Message("Applying LSI model"):
            # Transform X = tfidf_corpus from TF-IDF to LSI space. For X = U*S*V^T, this computes U^-1*X = V*S.
            # This reduces dimensions and "squishes" similar "topics"/related symbols into the same dimension.
            lsi_corpus = lsi_model[tfidf_corpus]

        if VERBOSE:
            U = lsi_model.projection.u # relates terms (rows) to topics (columns)
            S = lsi_model.projection.s # ranks relevance of topics
            V = corpus2dense(lsi_corpus, len(S)).T / S # relates documents (rows) to topics (vectors)
            print(f"U = {U.shape}, S = {S.shape}, V = {V.shape}")
            print(S)

        with Message("Storing index"):
            # Builds an index which we can compare queries against.
            index = Similarity(get_tmpfile(f"irsel_index"), lsi_corpus, num_features=len(dictionary))
        print(index)

        # TODO
        with Message("Querying for conjecture"):
            tokenized_conjecture = self.tokenize_formula(problem.conjecture)
            query = lsi_model[tfidf_model[dictionary.doc2bow(tokenized_conjecture)]]
            similarities = index[query]
            ranked_similarities = sorted(enumerate(similarities), key=lambda item: -item[1])
            top_matches = map(lambda elem: (problem.premises[elem[0]], elem[1]),
                map(lambda elem: elem[1],
                # arbitrary stop condition
                takewhile(lambda elem: elem[0] < 20 and elem[1][1] >= 1e-10, enumerate(ranked_similarities))))
        # possibly iterate?

        if VERBOSE:
            print("Selected axioms:")
            top_matches, top_matches_tmp = tee(top_matches)
            for premise, score in top_matches_tmp:
                print(f"{score}\t{premise.formula}")

        return Problem(premises=list(map(lambda elem: elem[0], top_matches)), conjecture=problem.conjecture) # TODO

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
            global VERBOSE
            VERBOSE = True
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
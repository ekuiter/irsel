#!/usr/bin/python3

from typing import Tuple, Iterable, Callable
from itertools import takewhile
from functools import partial, reduce

from gavel.selection.selector import Selector
from gavel.logic.problem import Problem, Sentence

from gensim.corpora import Dictionary, MmCorpus
from gensim.models import TfidfModel, LsiModel
from gensim.similarities import Similarity
from gensim.test.utils import get_tmpfile
from gensim.matutils import corpus2dense

from helpers import *

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

            if get_verbose():
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

        with Message("Selecting best-matching axioms", show_done=not get_verbose()):
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
                    if get_verbose() else identity, # print summary
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
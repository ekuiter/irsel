from typing import List, Tuple, Iterable
from functools import reduce
from timeit import default_timer
from datetime import timedelta

from gavel.selection.selector import Selector

# global variables
_verbose = False
_quiet = False

# constants
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
printq   = lambda *args, **kwargs: print(*args, **kwargs) if not _quiet else None

def set_verbose(verbose):
    global _verbose
    _verbose = verbose

def get_verbose():
    global _verbose
    return _verbose

def set_quiet(quiet):
    global _quiet
    _quiet = quiet

def get_quiet():
    global _quiet
    return _quiet

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
import random

def argmin_random_tie(seq, key=lambda x: x):
    """Return a minimum element of seq; break ties at random."""
    return min(shuffled(seq), key=key)

def count(seq):
    """Count the number of items in sequence that are interpreted as true."""
    return sum(map(bool, seq))

def first(iterable, default=None):
    """Return the first element of an iterable; or default."""
    return next(iter(iterable), default)

def shuffled(iterable):
    """Randomly shuffle a copy of iterable."""
    items = list(iterable)
    random.shuffle(items)
    return items

def is_in(elt, seq):
    """Similar to (elt in seq), but compares with 'is', not '=='."""
    return any(x is elt for x in seq)

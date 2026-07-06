from typing import Iterable, Iterator, TypeVar
from itertools import islice

T = TypeVar("T")

def batched(iterable: Iterable[T], n: int) -> Iterator[list[T]]:
    """
    Splits an iterable into batches of a specified size.
    
    Args:
        iterable (Iterable[T]): The input iterable to be batched.
        batch_size (int): The size of each batch.
        
    Yields:
        Iterator[List[T]]: An iterator of lists, each list containing a batch of items.
    """
    # Convert the iterable to an iterator
    iterator = iter(iterable)
    
    # Use iter and islice to yield batches of the specified size
    while True:
        batch = list(islice(iterator, n))
        if not batch:
            break
        yield batch
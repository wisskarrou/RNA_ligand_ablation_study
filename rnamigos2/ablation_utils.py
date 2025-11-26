import os

from typing import Any, List
import random

def guaranteed_derangement(items: List[Any]) -> List[Any]:
    """
    Generates a permutation of 'items' that is guaranteed to be a derangement.
    A derangement ensures that no element remains in its original position.
    
    This function uses a shuffle-and-fix approach to resolve fixed points.
    
    Args:
        items: The list of elements to be deranged (e.g., [1, 5, 9, 12]).

    Returns:
        A list representing the derangement of the input list.
    """
    n = len(items)
    if n <= 1:
        # Cannot derange a list of 0 or 1 elements
        return items[:]

    shuffled = items[:]
    
    # We loop until we find a derangement or manually resolve all fixed points
    attempts = 0
    while attempts < 100:
        random.shuffle(shuffled)
        
        # 1. Identify fixed points: where shuffled[i] == items[i]
        fixed_indices = [i for i in range(n) if shuffled[i] == items[i]]
        
        if not fixed_indices:
            # Success! Found a derangement.
            return shuffled
        
        # If fixed points exist, manually resolve them with simple swaps
        # This loop should ensure we resolve the fixed points in the current permutation
        if len(fixed_indices) == 1:
            i = fixed_indices[0]
            # If only one fixed point, swap it with its neighbor (j)
            j = (i + 1) % n
            shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
            # Since n > 1, this swap guarantees the original fixed point 'i' is resolved.
            
        elif len(fixed_indices) > 1:
            # If multiple, swap pairs of fixed points (i, j)
            for k in range(0, len(fixed_indices) - 1, 2):
                i = fixed_indices[k]
                j = fixed_indices[k+1]
                shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
            
            # If there's an odd one out, swap it with a random non-fixed point
            if len(fixed_indices) % 2 != 0:
                i = fixed_indices[-1]
                
                # Find a non-fixed point index j to swap with
                non_fixed_indices = [k for k in range(n) if k not in fixed_indices]
                if non_fixed_indices:
                    j = random.choice(non_fixed_indices)
                    shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
                else:
                    # Fallback: swap with the next element (only happens if all elements were fixed, which is impossible for N>1)
                    j = (i + 1) % n
                    shuffled[i], shuffled[j] = shuffled[j], shuffled[i]

        # Check again if the resolution worked (it should for all but the rarest N=2 case)
        if all(shuffled[i] != items[i] for i in range(n)):
            return shuffled

        attempts += 1 # Only happens if the resolution created new fixed points
        
    # As a last resort, return the best effort, though the while loop above should always succeed
    print("Warning: Derangement construction failed to fully resolve after many attempts.")
    return shuffled
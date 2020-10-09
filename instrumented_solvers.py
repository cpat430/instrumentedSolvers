from textbook.backtracking_search_solver import *
from textbook.min_conflicts_solver import *

#####################
# Your imports here #
#####################
import time


def backtracking_search_instrumented(
        csp,
        select_unassigned_variable=first_unassigned_variable,
        order_domain_values=unordered_domain_values,
        inference=no_inference,
        max_steps=100_000,
    ):
    """Return a dict where the key 'assignment' is identical to to result of backtracking_search(csp, select_unassigned_variable, ...).

    The key 'num_assignments' == the number of times csp.assign is called.
    The key 'num_backtracks' == the number of backtracks performed.
    """

    result = {}
    num_backtracks = 0

    class MaxSteps(Exception):
        """Raise to terminate backtracking."""

    def backtrack(assignment):
        nonlocal num_backtracks

        if len(assignment) == len(csp.variables):
            return assignment
        var = select_unassigned_variable(assignment, csp)
        for value in order_domain_values(var, assignment, csp):
            if csp.nassigns == max_steps:
                raise MaxSteps()
            if 0 == csp.nconflicts(var, value, assignment):
                csp.assign(var, value, assignment)
                removals = csp.suppose(var, value)
                if inference(csp, var, value, assignment, removals):
                    result = backtrack(assignment)
                    if result is not None:
                        return result
                    num_backtracks += 1
                csp.restore(removals)
        csp.unassign(var, assignment)
        return None

    try:

        # start backtracking from an empty state
        start = time.time()
        result = backtrack({})
    except MaxSteps:

        # if max steps is raised, then there is no solution?
        result = None
    
    end = time.time()
    # check if the result is valid, so empty or full.
    assert result is None or csp.goal_test(result)

    return {
        "assignment": result,
        "num_assignments": csp.nassigns,
        "num_backtracks": num_backtracks,
        "time": end - start,
    }




def min_conflicts_instrumented(csp, max_steps=100_000):
    """Return a dict where the key 'assignment' is identical to to result of min_conflicts(csp, max_steps).

    The key 'num_assignments' == the number of times csp.assign is called.
    The key 'num_repair_assignments' == the number of assignments made outside of generating the initial assignment of variables.
    """

    num_repair_assignments = 0

    def min_conflicts(max_steps = 100_000): # default value at 100,000
        nonlocal num_repair_assignments
        """Solve a CSP by stochastic Hill Climbing on the number of conflicts."""
        # Generate a complete assignment for all variables (probably with conflicts)
        csp.current = current = {}
        
        # initial assignment of variables
        for var in csp.variables:
            val = min_conflicts_value(csp, var, current)
            csp.assign(var, val, current)
            num_repair_assignments += 1

        
        # Now repeatedly choose a random conflicted variable and change it
        for i in range(max_steps):

            # check if the current state is conflicted
            conflicted = csp.conflicted_vars(current)
            if not conflicted:

                # returns the finished state
                return current

            # otherwise will choose the next conflicted thing
            var = random.choice(conflicted)

            # find the minimum conflicts value
            val = min_conflicts_value(csp, var, current)

            # assign the queen to the new minimum conflicts state
            csp.assign(var, val, current)
        
        # if no current found, return None
        return None

    # call to the function to find the solution state
    start = time.time()
    assignment = min_conflicts(max_steps)
    end = time.time()
    num_repair_assignments = csp.nassigns - num_repair_assignments

    return {
        "assignment": assignment,
        "num_assignments": csp.nassigns,
        "num_repair_assignments": num_repair_assignments,
        "time": end - start,
    }
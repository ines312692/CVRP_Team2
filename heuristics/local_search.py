"""
Local Search Algorithms for CVRP
Includes: 2-opt, Or-opt, Relocate, Exchange, Cross-exchange
Team 2 - Soft Computing Contest
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Any, Tuple, Optional
import copy
from common.cost import compute_solution_cost, compute_route_cost
from common.feasibility import is_feasible, get_route_load


# =============================================================================
# NEIGHBORHOOD OPERATORS
# =============================================================================

def two_opt_single_route(route: List[int], dm: Dict[int, Dict[int, float]]) -> Tuple[List[int], float]:
    """
    Apply 2-opt improvement to a single route.
    Reverses segments to eliminate crossings.

    Args:
        route: Route to improve [depot, c1, c2, ..., depot]
        dm: Distance matrix

    Returns:
        Tuple of (improved_route, improvement)
    """
    improved = True
    best_route = route[:]
    total_improvement = 0.0

    while improved:
        improved = False
        n = len(best_route)

        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                # Cost before: d(i-1,i) + d(j,j+1)
                # Cost after: d(i-1,j) + d(i,j+1)
                old_cost = dm[best_route[i - 1]][best_route[i]] + dm[best_route[j]][best_route[j + 1]]
                new_cost = dm[best_route[i - 1]][best_route[j]] + dm[best_route[i]][best_route[j + 1]]

                if new_cost < old_cost - 1e-10:
                    # Reverse segment [i, j]
                    best_route[i:j + 1] = best_route[i:j + 1][::-1]
                    total_improvement += old_cost - new_cost
                    improved = True
                    break
            if improved:
                break

    return best_route, total_improvement


def two_opt(solution: List[List[int]], instance: Dict[str, Any]) -> List[List[int]]:
    """
    Apply 2-opt to all routes in the solution.

    Args:
        solution: List of routes
        instance: Instance dictionary

    Returns:
        Improved solution
    """
    dm = instance['distance_matrix']
    improved_solution = []

    for route in solution:
        if len(route) > 3:  # At least one customer + depot at ends
            improved_route, _ = two_opt_single_route(route, dm)
            improved_solution.append(improved_route)
        else:
            improved_solution.append(route[:])

    return improved_solution


def or_opt(solution: List[List[int]], instance: Dict[str, Any],
           segment_sizes: List[int] = [1, 2, 3]) -> List[List[int]]:
    """
    Or-opt: Relocate segments of 1, 2, or 3 consecutive customers.

    Args:
        solution: List of routes
        instance: Instance dictionary
        segment_sizes: Sizes of segments to try relocating

    Returns:
        Improved solution
    """
    dm = instance['distance_matrix']
    demands = instance['demands']
    capacity = instance['capacity']
    depot = instance['depot']

    best_solution = [route[:] for route in solution]
    improved = True

    while improved:
        improved = False
        best_cost = compute_solution_cost(best_solution, dm)

        for seg_size in segment_sizes:
            for r1_idx, route1 in enumerate(best_solution):
                if len(route1) - 2 < seg_size:  # Not enough customers
                    continue

                # Try removing segment from route1
                for i in range(1, len(route1) - seg_size):
                    segment = route1[i:i + seg_size]
                    segment_demand = sum(demands.get(c, 0) for c in segment)

                    # New route1 without segment
                    new_route1 = route1[:i] + route1[i + seg_size:]

                    # Try inserting in all routes (including same route)
                    for r2_idx, route2 in enumerate(best_solution):
                        if r1_idx == r2_idx:
                            base_route2 = new_route1
                        else:
                            base_route2 = route2
                            # Check capacity
                            if (get_route_load(base_route2, demands, depot) +
                                    segment_demand > capacity):
                                continue

                        # Try all insertion positions
                        for j in range(1, len(base_route2)):
                            if r1_idx == r2_idx and abs(i - j) <= 1:
                                continue

                            # Create new route2 with segment
                            new_route2 = base_route2[:j] + segment + base_route2[j:]

                            # Build new solution
                            new_solution = best_solution[:]
                            if r1_idx == r2_idx:
                                new_solution[r1_idx] = new_route2
                            else:
                                new_solution[r1_idx] = new_route1
                                new_solution[r2_idx] = new_route2

                            # Remove empty routes
                            new_solution = [r for r in new_solution if len(r) > 2]

                            new_cost = compute_solution_cost(new_solution, dm)
                            if new_cost < best_cost - 1e-10:
                                best_solution = new_solution
                                best_cost = new_cost
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    return best_solution


def relocate(solution: List[List[int]], instance: Dict[str, Any]) -> List[List[int]]:
    """
    Relocate operator: Move a single customer to another position.

    Args:
        solution: List of routes
        instance: Instance dictionary

    Returns:
        Improved solution
    """
    return or_opt(solution, instance, segment_sizes=[1])


def exchange(solution: List[List[int]], instance: Dict[str, Any]) -> List[List[int]]:
    """
    Exchange operator: Swap two customers between routes.

    Args:
        solution: List of routes
        instance: Instance dictionary

    Returns:
        Improved solution
    """
    dm = instance['distance_matrix']
    demands = instance['demands']
    capacity = instance['capacity']
    depot = instance['depot']

    best_solution = [route[:] for route in solution]
    improved = True

    while improved:
        improved = False
        best_cost = compute_solution_cost(best_solution, dm)

        for r1_idx in range(len(best_solution)):
            for r2_idx in range(r1_idx, len(best_solution)):
                route1 = best_solution[r1_idx]
                route2 = best_solution[r2_idx]

                for i in range(1, len(route1) - 1):
                    j_start = i + 1 if r1_idx == r2_idx else 1
                    for j in range(j_start, len(route2) - 1):
                        c1 = route1[i]
                        c2 = route2[j]

                        if r1_idx != r2_idx:
                            # Check capacity after exchange
                            load1 = get_route_load(route1, demands, depot) - demands.get(c1, 0) + demands.get(c2, 0)
                            load2 = get_route_load(route2, demands, depot) - demands.get(c2, 0) + demands.get(c1, 0)

                            if load1 > capacity or load2 > capacity:
                                continue

                        # Create new routes
                        new_route1 = route1[:]
                        new_route2 = route2[:]
                        new_route1[i] = c2
                        new_route2[j] = c1

                        # Build new solution
                        new_solution = best_solution[:]
                        new_solution[r1_idx] = new_route1
                        if r1_idx != r2_idx:
                            new_solution[r2_idx] = new_route2

                        new_cost = compute_solution_cost(new_solution, dm)
                        if new_cost < best_cost - 1e-10:
                            best_solution = new_solution
                            best_cost = new_cost
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    return best_solution


def cross_exchange(solution: List[List[int]], instance: Dict[str, Any]) -> List[List[int]]:
    """
    Cross-exchange: Exchange tails between two routes.

    Args:
        solution: List of routes
        instance: Instance dictionary

    Returns:
        Improved solution
    """
    dm = instance['distance_matrix']
    demands = instance['demands']
    capacity = instance['capacity']
    depot = instance['depot']

    best_solution = [route[:] for route in solution]
    improved = True

    while improved:
        improved = False
        best_cost = compute_solution_cost(best_solution, dm)

        for r1_idx in range(len(best_solution) - 1):
            for r2_idx in range(r1_idx + 1, len(best_solution)):
                route1 = best_solution[r1_idx]
                route2 = best_solution[r2_idx]

                for i in range(1, len(route1) - 1):
                    for j in range(1, len(route2) - 1):
                        # Exchange tails: route1[i:] with route2[j:]
                        tail1 = route1[i:-1]
                        tail2 = route2[j:-1]

                        # New routes
                        new_route1 = route1[:i] + tail2 + [depot]
                        new_route2 = route2[:j] + tail1 + [depot]

                        # Check capacity
                        if (get_route_load(new_route1, demands, depot) > capacity or
                                get_route_load(new_route2, demands, depot) > capacity):
                            continue

                        new_solution = best_solution[:]
                        new_solution[r1_idx] = new_route1
                        new_solution[r2_idx] = new_route2

                        new_cost = compute_solution_cost(new_solution, dm)
                        if new_cost < best_cost - 1e-10:
                            best_solution = new_solution
                            best_cost = new_cost
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    return best_solution


# =============================================================================
# COMBINED LOCAL SEARCH
# =============================================================================

def local_search(solution: List[List[int]], instance: Dict[str, Any],
                 operators: Optional[List[str]] = None,
                 max_iterations: int = 1000) -> Tuple[List[List[int]], Dict[str, Any]]:
    """
    Combined local search using multiple operators.

    Args:
        solution: Initial solution
        instance: Instance dictionary
        operators: List of operators to use ['2opt', 'relocate', 'exchange', 'oropt', 'cross']
        max_iterations: Maximum iterations without improvement

    Returns:
        Tuple of (best_solution, search_info)
    """
    if operators is None:
        operators = ['2opt', 'relocate', 'exchange', 'oropt']

    dm = instance['distance_matrix']

    operator_funcs = {
        '2opt': two_opt,
        'relocate': relocate,
        'exchange': exchange,
        'oropt': or_opt,
        'cross': cross_exchange
    }

    best_solution = [route[:] for route in solution]
    best_cost = compute_solution_cost(best_solution, dm)

    info = {
        'iterations': 0,
        'improvements': 0,
        'initial_cost': best_cost,
        'operator_improvements': {op: 0 for op in operators}
    }

    no_improvement_count = 0

    while no_improvement_count < max_iterations:
        improved = False

        for op_name in operators:
            if op_name not in operator_funcs:
                continue

            new_solution = operator_funcs[op_name](best_solution, instance)
            new_cost = compute_solution_cost(new_solution, dm)

            if new_cost < best_cost - 1e-10:
                best_solution = new_solution
                best_cost = new_cost
                improved = True
                info['improvements'] += 1
                info['operator_improvements'][op_name] += 1
                no_improvement_count = 0
                break

        if not improved:
            no_improvement_count += 1

        info['iterations'] += 1

    info['final_cost'] = best_cost
    info['total_improvement'] = info['initial_cost'] - best_cost

    return best_solution, info


def variable_neighborhood_descent(solution: List[List[int]],
                                  instance: Dict[str, Any],
                                  neighborhoods: Optional[List[str]] = None) -> List[List[int]]:
    """
    Variable Neighborhood Descent (VND).
    Systematically explores neighborhoods in a fixed order.

    Args:
        solution: Initial solution
        instance: Instance dictionary
        neighborhoods: List of neighborhood operators

    Returns:
        Improved solution
    """
    if neighborhoods is None:
        neighborhoods = ['2opt', 'relocate', 'exchange', 'oropt', 'cross']

    dm = instance['distance_matrix']

    operator_funcs = {
        '2opt': two_opt,
        'relocate': relocate,
        'exchange': exchange,
        'oropt': or_opt,
        'cross': cross_exchange
    }

    current_solution = [route[:] for route in solution]
    current_cost = compute_solution_cost(current_solution, dm)

    k = 0
    while k < len(neighborhoods):
        op_name = neighborhoods[k]
        if op_name not in operator_funcs:
            k += 1
            continue

        new_solution = operator_funcs[op_name](current_solution, instance)
        new_cost = compute_solution_cost(new_solution, dm)

        if new_cost < current_cost - 1e-10:
            current_solution = new_solution
            current_cost = new_cost
            k = 0  # Restart from first neighborhood
        else:
            k += 1  # Move to next neighborhood

    return current_solution


# =============================================================================
# MAIN FUNCTION
# =============================================================================

if __name__ == '__main__':
    from common.reader import read_instance
    from common.utils import Timer, print_result, create_result
    from heuristics.greedy import nearest_neighbor

    print("Local Search for CVRP")
    print("=" * 50)

    # Create test instance
    instance = {
        'name': 'test',
        'depot': 1,
        'capacity': 100,
        'demands': {1: 0, 2: 20, 3: 30, 4: 40, 5: 25, 6: 35, 7: 15, 8: 30},
        'coordinates': {
            1: (50, 50), 2: (20, 30), 3: (40, 80), 4: (70, 60),
            5: (30, 50), 6: (60, 30), 7: (80, 70), 8: (25, 65)
        },
        'optimal': None
    }

    # Compute distance matrix
    from common.cost import compute_distance_matrix

    instance['distance_matrix'] = {}
    for i in instance['coordinates']:
        instance['distance_matrix'][i] = {}
        for j in instance['coordinates']:
            xi, yi = instance['coordinates'][i]
            xj, yj = instance['coordinates'][j]
            instance['distance_matrix'][i][j] = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5

    # Get initial solution
    initial = nearest_neighbor(instance)
    initial_cost = compute_solution_cost(initial, instance['distance_matrix'])
    print(f"\nInitial solution (Nearest Neighbor): {initial_cost:.2f}")

    # Test each operator
    print("\nTesting individual operators:")
    print("-" * 50)

    operators = [
        ('2-opt', two_opt),
        ('Relocate', relocate),
        ('Exchange', exchange),
        ('Or-opt', or_opt),
        ('Cross-exchange', cross_exchange)
    ]

    for name, func in operators:
        with Timer() as t:
            improved = func([route[:] for route in initial], instance)
        cost = compute_solution_cost(improved, instance['distance_matrix'])
        improvement = initial_cost - cost
        print(f"{name:<20}: {cost:>10.2f} (improvement: {improvement:>8.2f}, time: {t.elapsed:.4f}s)")

    # Test combined local search
    print("\nCombined Local Search:")
    print("-" * 50)

    with Timer() as t:
        final, info = local_search([route[:] for route in initial], instance)

    print(f"Final cost: {info['final_cost']:.2f}")
    print(f"Total improvement: {info['total_improvement']:.2f}")
    print(f"Iterations: {info['iterations']}")
    print(f"Time: {t.elapsed:.4f}s")
    print(f"\nOperator improvements: {info['operator_improvements']}")

    # Test VND
    print("\nVariable Neighborhood Descent:")
    print("-" * 50)

    with Timer() as t:
        vnd_solution = variable_neighborhood_descent([route[:] for route in initial], instance)

    vnd_cost = compute_solution_cost(vnd_solution, instance['distance_matrix'])
    print(f"VND cost: {vnd_cost:.2f}")
    print(f"Time: {t.elapsed:.4f}s")
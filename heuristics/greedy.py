"""
Greedy Heuristics for CVRP
Includes: Nearest Neighbor, Savings Algorithm, Sequential Insertion
Team 2 - Soft Computing Contest
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Any, Tuple, Optional
from common.cost import compute_solution_cost, compute_route_cost
from common.feasibility import is_feasible, get_route_load, can_insert_customer


# =============================================================================
# NEAREST NEIGHBOR HEURISTIC
# =============================================================================

def nearest_neighbor(instance: Dict[str, Any], start_from_depot: bool = True) -> List[List[int]]:
    """
    Nearest Neighbor heuristic for CVRP.
    Builds routes by always visiting the nearest unvisited customer.

    Args:
        instance: Instance dictionary
        start_from_depot: If True, always start from depot

    Returns:
        Solution as list of routes
    """
    depot = instance['depot']
    capacity = instance['capacity']
    demands = instance['demands']
    dm = instance['distance_matrix']

    # Get all customers (exclude depot)
    unvisited = set(node for node in instance['coordinates'].keys() if node != depot)

    solution = []

    while unvisited:
        # Start new route
        route = [depot]
        current_load = 0
        current_node = depot

        while unvisited:
            # Find nearest feasible customer
            best_customer = None
            best_distance = float('inf')

            for customer in unvisited:
                demand = demands.get(customer, 0)
                if current_load + demand <= capacity:
                    dist = dm[current_node][customer]
                    if dist < best_distance:
                        best_distance = dist
                        best_customer = customer

            if best_customer is None:
                # No more customers can be added to this route
                break

            # Add customer to route
            route.append(best_customer)
            current_load += demands.get(best_customer, 0)
            current_node = best_customer
            unvisited.remove(best_customer)

        # Close route
        route.append(depot)
        solution.append(route)

    return solution


def nearest_neighbor_enhanced(instance: Dict[str, Any]) -> List[List[int]]:
    """
    Enhanced Nearest Neighbor that considers both distance and demand.
    Prioritizes customers with high demand-to-distance ratio.

    Args:
        instance: Instance dictionary

    Returns:
        Solution as list of routes
    """
    depot = instance['depot']
    capacity = instance['capacity']
    demands = instance['demands']
    dm = instance['distance_matrix']

    unvisited = set(node for node in instance['coordinates'].keys() if node != depot)
    solution = []

    while unvisited:
        route = [depot]
        current_load = 0
        current_node = depot

        while unvisited:
            best_customer = None
            best_score = float('-inf')

            for customer in unvisited:
                demand = demands.get(customer, 0)
                if current_load + demand <= capacity:
                    dist = dm[current_node][customer]
                    if dist > 0:
                        # Score: prioritize high demand relative to distance
                        score = demand / dist
                    else:
                        score = float('inf')

                    if score > best_score:
                        best_score = score
                        best_customer = customer

            if best_customer is None:
                break

            route.append(best_customer)
            current_load += demands.get(best_customer, 0)
            current_node = best_customer
            unvisited.remove(best_customer)

        route.append(depot)
        solution.append(route)

    return solution


# =============================================================================
# SAVINGS ALGORITHM (Clarke-Wright)
# =============================================================================

def savings_algorithm(instance: Dict[str, Any], parallel: bool = True) -> List[List[int]]:
    """
    Clarke-Wright Savings Algorithm for CVRP.
    Merges routes based on savings from eliminating depot visits.

    Args:
        instance: Instance dictionary
        parallel: If True, use parallel version (consider all merges)

    Returns:
        Solution as list of routes
    """
    depot = instance['depot']
    capacity = instance['capacity']
    demands = instance['demands']
    dm = instance['distance_matrix']

    customers = [node for node in instance['coordinates'].keys() if node != depot]

    # Initialize: each customer in its own route
    routes = {i: [depot, c, depot] for i, c in enumerate(customers)}
    customer_to_route = {c: i for i, c in enumerate(customers)}

    # Compute savings: s(i,j) = d(depot,i) + d(depot,j) - d(i,j)
    savings = []
    for i, ci in enumerate(customers):
        for j, cj in enumerate(customers):
            if i < j:
                saving = dm[depot][ci] + dm[depot][cj] - dm[ci][cj]
                savings.append((saving, ci, cj))

    # Sort by decreasing savings
    savings.sort(reverse=True, key=lambda x: x[0])

    # Merge routes
    for saving, ci, cj in savings:
        if saving <= 0:
            break

        route_i = customer_to_route.get(ci)
        route_j = customer_to_route.get(cj)

        if route_i is None or route_j is None or route_i == route_j:
            continue

        if route_i not in routes or route_j not in routes:
            continue

        # Get routes
        r_i = routes[route_i]
        r_j = routes[route_j]

        # Check if customers are at route endpoints
        ci_at_end = (r_i[-2] == ci)  # ci is last customer before depot
        ci_at_start = (r_i[1] == ci)  # ci is first customer after depot
        cj_at_end = (r_j[-2] == cj)
        cj_at_start = (r_j[1] == cj)

        # Only merge if customers are at endpoints
        if not ((ci_at_end or ci_at_start) and (cj_at_end or cj_at_start)):
            continue

        # Check capacity
        load_i = get_route_load(r_i, demands, depot)
        load_j = get_route_load(r_j, demands, depot)

        if load_i + load_j > capacity:
            continue

        # Merge routes
        if ci_at_end and cj_at_start:
            # r_i: [depot, ..., ci, depot] + r_j: [depot, cj, ..., depot]
            new_route = r_i[:-1] + r_j[1:]
        elif ci_at_start and cj_at_end:
            # r_j: [depot, ..., cj, depot] + r_i: [depot, ci, ..., depot]
            new_route = r_j[:-1] + r_i[1:]
        elif ci_at_end and cj_at_end:
            # Reverse r_j and append
            r_j_reversed = [depot] + r_j[-2:0:-1] + [depot]
            new_route = r_i[:-1] + r_j_reversed[1:]
        elif ci_at_start and cj_at_start:
            # Reverse r_i and prepend
            r_i_reversed = [depot] + r_i[-2:0:-1] + [depot]
            new_route = r_i_reversed[:-1] + r_j[1:]
        else:
            continue

        # Update routes
        routes[route_i] = new_route
        del routes[route_j]

        # Update customer mapping
        for c in new_route:
            if c != depot:
                customer_to_route[c] = route_i

    return list(routes.values())


def savings_algorithm_sequential(instance: Dict[str, Any]) -> List[List[int]]:
    """
    Sequential version of Savings Algorithm.
    Extends one route at a time until no more savings possible.

    Args:
        instance: Instance dictionary

    Returns:
        Solution as list of routes
    """
    depot = instance['depot']
    capacity = instance['capacity']
    demands = instance['demands']
    dm = instance['distance_matrix']

    customers = set(node for node in instance['coordinates'].keys() if node != depot)
    solution = []

    while customers:
        # Start with the farthest unvisited customer
        farthest = max(customers, key=lambda c: dm[depot][c])
        route = [depot, farthest, depot]
        current_load = demands.get(farthest, 0)
        customers.remove(farthest)

        improved = True
        while improved and customers:
            improved = False
            best_saving = 0
            best_customer = None
            best_position = None

            for customer in customers:
                demand = demands.get(customer, 0)
                if current_load + demand > capacity:
                    continue

                # Try inserting at start (after depot)
                saving_start = (dm[depot][route[1]] -
                                (dm[depot][customer] + dm[customer][route[1]]))
                if -saving_start > best_saving:
                    best_saving = -saving_start
                    best_customer = customer
                    best_position = 1

                # Try inserting at end (before depot)
                saving_end = (dm[route[-2]][depot] -
                              (dm[route[-2]][customer] + dm[customer][depot]))
                if -saving_end > best_saving:
                    best_saving = -saving_end
                    best_customer = customer
                    best_position = len(route) - 1

            if best_customer is not None:
                route.insert(best_position, best_customer)
                current_load += demands.get(best_customer, 0)
                customers.remove(best_customer)
                improved = True

        solution.append(route)

    return solution


# =============================================================================
# SEQUENTIAL INSERTION HEURISTIC
# =============================================================================

def sequential_insertion(instance: Dict[str, Any],
                         criterion: str = 'cheapest') -> List[List[int]]:
    """
    Sequential Insertion heuristic for CVRP.
    Inserts customers one by one in the best position.

    Args:
        instance: Instance dictionary
        criterion: 'cheapest' (minimize cost increase) or 'nearest' (to route)

    Returns:
        Solution as list of routes
    """
    depot = instance['depot']
    capacity = instance['capacity']
    demands = instance['demands']
    dm = instance['distance_matrix']

    customers = list(node for node in instance['coordinates'].keys() if node != depot)
    # Sort by distance to depot (farthest first)
    customers.sort(key=lambda c: dm[depot][c], reverse=True)

    solution = []
    unassigned = set(customers)

    while unassigned:
        # Start new route with farthest unassigned customer
        seed = max(unassigned, key=lambda c: dm[depot][c])
        route = [depot, seed, depot]
        current_load = demands.get(seed, 0)
        unassigned.remove(seed)

        improved = True
        while improved and unassigned:
            improved = False
            best_cost = float('inf')
            best_customer = None
            best_pos = None

            for customer in unassigned:
                demand = demands.get(customer, 0)
                if current_load + demand > capacity:
                    continue

                # Find best position
                for pos in range(1, len(route)):
                    if criterion == 'cheapest':
                        # Cost of insertion
                        cost = (dm[route[pos - 1]][customer] +
                                dm[customer][route[pos]] -
                                dm[route[pos - 1]][route[pos]])
                    else:  # nearest
                        cost = min(dm[route[pos - 1]][customer],
                                   dm[customer][route[pos]])

                    if cost < best_cost:
                        best_cost = cost
                        best_customer = customer
                        best_pos = pos

            if best_customer is not None:
                route.insert(best_pos, best_customer)
                current_load += demands.get(best_customer, 0)
                unassigned.remove(best_customer)
                improved = True

        solution.append(route)

    return solution


# =============================================================================
# SWEEP ALGORITHM
# =============================================================================

def sweep_algorithm(instance: Dict[str, Any],
                    clockwise: bool = True) -> List[List[int]]:
    """
    Sweep Algorithm for CVRP.
    Groups customers by angular position relative to depot.

    Args:
        instance: Instance dictionary
        clockwise: Direction of sweep

    Returns:
        Solution as list of routes
    """
    import math

    depot = instance['depot']
    capacity = instance['capacity']
    demands = instance['demands']
    coords = instance['coordinates']
    dm = instance['distance_matrix']

    depot_x, depot_y = coords[depot]

    # Compute angles for all customers
    customers_with_angles = []
    for node in coords:
        if node != depot:
            x, y = coords[node]
            angle = math.atan2(y - depot_y, x - depot_x)
            customers_with_angles.append((node, angle))

    # Sort by angle
    customers_with_angles.sort(key=lambda x: x[1], reverse=not clockwise)

    solution = []
    route = [depot]
    current_load = 0

    for customer, _ in customers_with_angles:
        demand = demands.get(customer, 0)

        if current_load + demand <= capacity:
            route.append(customer)
            current_load += demand
        else:
            # Close current route and start new one
            route.append(depot)
            solution.append(route)
            route = [depot, customer]
            current_load = demand

    # Close last route
    if len(route) > 1:
        route.append(depot)
        solution.append(route)

    return solution


# =============================================================================
# MAIN COMPARISON FUNCTION
# =============================================================================

def compare_greedy_heuristics(instance: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Compare all greedy heuristics on an instance.

    Args:
        instance: Instance dictionary

    Returns:
        Dictionary with results for each heuristic
    """
    from common.utils import Timer

    heuristics = {
        'Nearest Neighbor': nearest_neighbor,
        'Nearest Neighbor Enhanced': nearest_neighbor_enhanced,
        'Savings Parallel': lambda i: savings_algorithm(i, parallel=True),
        'Savings Sequential': savings_algorithm_sequential,
        'Sequential Insertion': sequential_insertion,
        'Sweep': sweep_algorithm
    }

    results = {}

    for name, func in heuristics.items():
        with Timer() as t:
            solution = func(instance)

        cost = compute_solution_cost(solution, instance['distance_matrix'])
        feasible = is_feasible(solution, instance)

        results[name] = {
            'solution': solution,
            'cost': cost,
            'feasible': feasible,
            'time': t.elapsed,
            'num_routes': len(solution)
        }

        if instance.get('optimal'):
            results[name]['gap'] = ((cost - instance['optimal']) /
                                    instance['optimal']) * 100

    return results


if __name__ == '__main__':
    from common.reader import read_instance, get_instance_info
    from common.utils import print_result, create_result

    # Test with command line argument or default
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        print("Usage: python greedy.py <instance_file>")
        print("Creating a test instance...")

        # Create test instance
        instance = {
            'name': 'test',
            'depot': 1,
            'capacity': 100,
            'demands': {1: 0, 2: 20, 3: 30, 4: 40, 5: 25, 6: 35},
            'coordinates': {
                1: (50, 50), 2: (20, 30), 3: (40, 80),
                4: (70, 60), 5: (30, 50), 6: (60, 30)
            },
            'optimal': None
        }

        from common.cost import compute_distance_matrix

        instance['distance_matrix'] = {}
        for i in instance['coordinates']:
            instance['distance_matrix'][i] = {}
            for j in instance['coordinates']:
                xi, yi = instance['coordinates'][i]
                xj, yj = instance['coordinates'][j]
                instance['distance_matrix'][i][j] = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5

        print("\nComparing greedy heuristics:")
        results = compare_greedy_heuristics(instance)

        print(f"\n{'Heuristic':<30} {'Cost':>10} {'Routes':>8} {'Time':>10} {'Feasible':>10}")
        print("-" * 70)
        for name, res in sorted(results.items(), key=lambda x: x[1]['cost']):
            print(f"{name:<30} {res['cost']:>10.2f} {res['num_routes']:>8} "
                  f"{res['time']:>10.4f}s {'Yes' if res['feasible'] else 'No':>10}")
"""
CVRP Cost Computation Module
Calculates the total cost (distance) of a CVRP solution
Team 1 - Soft Computing Contest
"""

from typing import Dict, List, Any


def compute_route_cost(route: List[int], distance_matrix: Dict[int, Dict[int, float]]) -> float:
    """
    Compute the total distance of a single route.

    Args:
        route: List of nodes representing a route [depot, c1, c2, ..., depot]
        distance_matrix: 2D dict of distances between nodes

    Returns:
        Total distance of the route
    """
    if len(route) < 2:
        return 0.0

    total = 0.0
    for i in range(len(route) - 1):
        total += distance_matrix[route[i]][route[i + 1]]

    return total


def compute_solution_cost(solution: List[List[int]], distance_matrix: Dict[int, Dict[int, float]]) -> float:
    """
    Compute the total cost of a CVRP solution.

    Args:
        solution: List of routes, each route is [depot, c1, c2, ..., depot]
        distance_matrix: 2D dict of distances between nodes

    Returns:
        Total cost (sum of all route distances)
    """
    total_cost = 0.0
    for route in solution:
        total_cost += compute_route_cost(route, distance_matrix)

    return total_cost


def compute_solution_cost_from_instance(solution: List[List[int]], instance: Dict[str, Any]) -> float:
    """
    Compute solution cost using instance dictionary.

    Args:
        solution: List of routes
        instance: Instance dictionary from reader

    Returns:
        Total solution cost
    """
    return compute_solution_cost(solution, instance['distance_matrix'])


def compute_route_load(route: List[int], demands: Dict[int, int], depot: int = 1) -> int:
    """
    Compute total demand served by a route.

    Args:
        route: List of nodes representing a route
        demands: Dict mapping node -> demand
        depot: Depot node (demand is 0)

    Returns:
        Total demand of customers in the route
    """
    total_load = 0
    for node in route:
        if node != depot:
            total_load += demands.get(node, 0)
    return total_load


def compute_gap(solution_cost: float, optimal: float) -> float:
    """
    Compute the gap between solution cost and optimal value.

    Args:
        solution_cost: Cost of the solution
        optimal: Known optimal value

    Returns:
        Gap as percentage: ((solution - optimal) / optimal) * 100
    """
    if optimal is None or optimal == 0:
        return None
    return ((solution_cost - optimal) / optimal) * 100


def get_cost_breakdown(solution: List[List[int]], instance: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get detailed cost breakdown for a solution.

    Args:
        solution: List of routes
        instance: Instance dictionary

    Returns:
        Dict with cost details per route and total
    """
    breakdown = {
        'routes': [],
        'total_cost': 0.0,
        'num_routes': len(solution),
        'num_vehicles': len(solution)
    }

    for i, route in enumerate(solution):
        route_cost = compute_route_cost(route, instance['distance_matrix'])
        route_load = compute_route_load(route, instance['demands'], instance['depot'])

        breakdown['routes'].append({
            'route_id': i + 1,
            'nodes': route,
            'cost': route_cost,
            'load': route_load,
            'num_customers': len(route) - 2  # Exclude depot at start and end
        })
        breakdown['total_cost'] += route_cost

    # Add gap if optimal is known
    if instance.get('optimal'):
        breakdown['gap'] = compute_gap(breakdown['total_cost'], instance['optimal'])
        breakdown['optimal'] = instance['optimal']

    return breakdown


if __name__ == '__main__':
    # Simple test
    dm = {
        1: {1: 0, 2: 10, 3: 15, 4: 20},
        2: {1: 10, 2: 0, 3: 12, 4: 8},
        3: {1: 15, 2: 12, 3: 0, 4: 10},
        4: {1: 20, 2: 8, 3: 10, 4: 0}
    }

    solution = [
        [1, 2, 4, 1],
        [1, 3, 1]
    ]

    cost = compute_solution_cost(solution, dm)
    print(f"Test solution cost: {cost}")
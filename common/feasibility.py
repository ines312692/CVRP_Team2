"""
CVRP Feasibility Checking Module
Validates CVRP solutions against all constraints
Team 1 - Soft Computing Contest
"""

from typing import Dict, List, Any, Tuple


def check_feasibility(solution: List[List[int]], instance: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Check if a solution is feasible for the given CVRP instance.

    Args:
        solution: List of routes, each route is [depot, c1, c2, ..., depot]
        instance: Instance dictionary from reader

    Returns:
        Tuple of (is_feasible, list_of_violations)
    """
    violations = []
    depot = instance['depot']
    capacity = instance['capacity']
    demands = instance['demands']
    customers = set(node for node in instance['coordinates'].keys() if node != depot)

    visited_customers = set()

    for route_idx, route in enumerate(solution):
        route_num = route_idx + 1

        # Check 1: Route starts and ends at depot
        if len(route) < 2:
            violations.append(f"Route {route_num}: Too short (must have at least depot-depot)")
            continue

        if route[0] != depot:
            violations.append(f"Route {route_num}: Does not start at depot (starts at {route[0]})")

        if route[-1] != depot:
            violations.append(f"Route {route_num}: Does not end at depot (ends at {route[-1]})")

        # Check 2: Capacity constraint
        route_load = 0
        for node in route:
            if node != depot:
                route_load += demands.get(node, 0)

        if route_load > capacity:
            violations.append(f"Route {route_num}: Capacity exceeded ({route_load} > {capacity})")

        # Check 3: Track visited customers
        for node in route:
            if node != depot:
                if node in visited_customers:
                    violations.append(f"Route {route_num}: Customer {node} visited more than once")
                else:
                    visited_customers.add(node)

    # Check 4: All customers must be visited
    unvisited = customers - visited_customers
    if unvisited:
        violations.append(f"Unvisited customers: {sorted(unvisited)}")

    # Check 5: No invalid customers
    invalid_customers = visited_customers - customers
    if invalid_customers:
        violations.append(f"Invalid customer nodes: {sorted(invalid_customers)}")

    is_feasible = len(violations) == 0
    return is_feasible, violations


def is_feasible(solution: List[List[int]], instance: Dict[str, Any]) -> bool:
    """
    Quick feasibility check.

    Args:
        solution: List of routes
        instance: Instance dictionary

    Returns:
        True if solution is feasible, False otherwise
    """
    feasible, _ = check_feasibility(solution, instance)
    return feasible


def check_route_capacity(route: List[int], demands: Dict[int, int],
                         capacity: int, depot: int = 1) -> bool:
    """
    Check if a single route respects capacity constraint.

    Args:
        route: List of nodes in the route
        demands: Dict mapping node -> demand
        capacity: Vehicle capacity
        depot: Depot node

    Returns:
        True if route load <= capacity
    """
    load = sum(demands.get(node, 0) for node in route if node != depot)
    return load <= capacity


def get_route_load(route: List[int], demands: Dict[int, int], depot: int = 1) -> int:
    """
    Get total load of a route.

    Args:
        route: List of nodes
        demands: Dict mapping node -> demand
        depot: Depot node

    Returns:
        Total demand served by the route
    """
    return sum(demands.get(node, 0) for node in route if node != depot)


def can_insert_customer(route: List[int], customer: int, demands: Dict[int, int],
                        capacity: int, depot: int = 1) -> bool:
    """
    Check if a customer can be inserted into a route without violating capacity.

    Args:
        route: Current route
        customer: Customer to potentially insert
        demands: Dict mapping node -> demand
        capacity: Vehicle capacity
        depot: Depot node

    Returns:
        True if customer can be inserted
    """
    current_load = get_route_load(route, demands, depot)
    customer_demand = demands.get(customer, 0)
    return current_load + customer_demand <= capacity


def get_feasibility_report(solution: List[List[int]], instance: Dict[str, Any]) -> str:
    """
    Get a detailed feasibility report.

    Args:
        solution: List of routes
        instance: Instance dictionary

    Returns:
        Formatted report string
    """
    feasible, violations = check_feasibility(solution, instance)

    report = f"""
=== Feasibility Report ===
Instance: {instance['name']}
Solution Status: {'FEASIBLE' if feasible else 'INFEASIBLE'}
Number of Routes: {len(solution)}
"""

    if not feasible:
        report += f"\nViolations Found ({len(violations)}):\n"
        for v in violations:
            report += f"  - {v}\n"
    else:
        report += "\nAll constraints satisfied:\n"
        report += "  ✓ All routes start and end at depot\n"
        report += "  ✓ All capacity constraints respected\n"
        report += "  ✓ All customers visited exactly once\n"

    # Route details
    report += "\nRoute Details:\n"
    depot = instance['depot']
    for i, route in enumerate(solution):
        load = get_route_load(route, instance['demands'], depot)
        report += f"  Route {i + 1}: {len(route) - 2} customers, Load: {load}/{instance['capacity']}\n"

    return report


if __name__ == '__main__':
    # Simple test
    instance = {
        'name': 'test',
        'depot': 1,
        'capacity': 100,
        'demands': {1: 0, 2: 30, 3: 40, 4: 50},
        'coordinates': {1: (0, 0), 2: (1, 1), 3: (2, 2), 4: (3, 3)}
    }

    # Feasible solution
    solution = [
        [1, 2, 3, 1],
        [1, 4, 1]
    ]

    print(get_feasibility_report(solution, instance))
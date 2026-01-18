"""
Tabu Search for CVRP
Team 2 - Soft Computing Contest
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Any, Tuple, Optional, Set
import copy
import random
from collections import deque
from common.cost import compute_solution_cost, compute_route_cost
from common.feasibility import is_feasible, get_route_load


class TabuSearch:
    """
    Tabu Search implementation for CVRP.

    Features:
    - Multiple neighborhood operators
    - Aspiration criteria
    - Diversification and intensification
    - Dynamic tabu tenure
    """

    def __init__(self, instance: Dict[str, Any],
                 tabu_tenure: int = 10,
                 max_iterations: int = 1000,
                 max_no_improve: int = 100,
                 aspiration: bool = True,
                 diversification: bool = True,
                 verbose: bool = False):
        """
        Initialize Tabu Search.

        Args:
            instance: CVRP instance
            tabu_tenure: Number of iterations a move stays tabu
            max_iterations: Maximum total iterations
            max_no_improve: Max iterations without improvement (stopping criterion)
            aspiration: Enable aspiration criteria
            diversification: Enable diversification
            verbose: Print progress
        """
        self.instance = instance
        self.depot = instance['depot']
        self.capacity = instance['capacity']
        self.demands = instance['demands']
        self.dm = instance['distance_matrix']

        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations
        self.max_no_improve = max_no_improve
        self.aspiration = aspiration
        self.diversification = diversification
        self.verbose = verbose

        # Tabu list: stores (move_type, customer, iteration_added)
        self.tabu_list = deque(maxlen=tabu_tenure * 10)

        # Statistics
        self.stats = {
            'iterations': 0,
            'improvements': 0,
            'aspiration_used': 0,
            'diversification_count': 0,
            'cost_history': []
        }

    def _is_tabu(self, move: Tuple, current_iter: int) -> bool:
        """Check if a move is tabu."""
        for tabu_move, added_iter in self.tabu_list:
            if tabu_move == move and current_iter - added_iter < self.tabu_tenure:
                return True
        return False

    def _add_tabu(self, move: Tuple, current_iter: int):
        """Add a move to the tabu list."""
        self.tabu_list.append((move, current_iter))

    def _get_relocate_moves(self, solution: List[List[int]]) -> List[Tuple]:
        """Generate all relocate moves."""
        moves = []

        for r1_idx, route1 in enumerate(solution):
            for i in range(1, len(route1) - 1):
                customer = route1[i]

                for r2_idx, route2 in enumerate(solution):
                    if r1_idx == r2_idx:
                        # Same route relocate
                        for j in range(1, len(route2)):
                            if abs(i - j) > 1:
                                moves.append(('relocate', customer, r1_idx, i, r2_idx, j))
                    else:
                        # Different route relocate
                        # Check capacity
                        customer_demand = self.demands.get(customer, 0)
                        route2_load = get_route_load(route2, self.demands, self.depot)

                        if route2_load + customer_demand <= self.capacity:
                            for j in range(1, len(route2)):
                                moves.append(('relocate', customer, r1_idx, i, r2_idx, j))

        return moves

    def _get_exchange_moves(self, solution: List[List[int]]) -> List[Tuple]:
        """Generate all exchange moves."""
        moves = []

        for r1_idx in range(len(solution)):
            for r2_idx in range(r1_idx, len(solution)):
                route1 = solution[r1_idx]
                route2 = solution[r2_idx]

                for i in range(1, len(route1) - 1):
                    j_start = i + 1 if r1_idx == r2_idx else 1
                    for j in range(j_start, len(route2) - 1):
                        c1 = route1[i]
                        c2 = route2[j]

                        if r1_idx != r2_idx:
                            # Check capacity
                            load1 = get_route_load(route1, self.demands, self.depot)
                            load2 = get_route_load(route2, self.demands, self.depot)
                            d1, d2 = self.demands.get(c1, 0), self.demands.get(c2, 0)

                            if load1 - d1 + d2 > self.capacity or load2 - d2 + d1 > self.capacity:
                                continue

                        moves.append(('exchange', c1, c2, r1_idx, i, r2_idx, j))

        return moves

    def _apply_relocate(self, solution: List[List[int]], move: Tuple) -> List[List[int]]:
        """Apply a relocate move."""
        _, customer, r1_idx, pos1, r2_idx, pos2 = move

        new_solution = [route[:] for route in solution]

        # Remove from original position
        new_solution[r1_idx].pop(pos1)

        # Adjust position if same route and after removal position
        if r1_idx == r2_idx and pos2 > pos1:
            pos2 -= 1

        # Insert at new position
        new_solution[r2_idx].insert(pos2, customer)

        # Remove empty routes
        new_solution = [r for r in new_solution if len(r) > 2]

        return new_solution

    def _apply_exchange(self, solution: List[List[int]], move: Tuple) -> List[List[int]]:
        """Apply an exchange move."""
        _, c1, c2, r1_idx, pos1, r2_idx, pos2 = move

        new_solution = [route[:] for route in solution]

        if r1_idx == r2_idx:
            new_solution[r1_idx][pos1] = c2
            new_solution[r1_idx][pos2] = c1
        else:
            new_solution[r1_idx][pos1] = c2
            new_solution[r2_idx][pos2] = c1

        return new_solution

    def _evaluate_move(self, solution: List[List[int]], move: Tuple) -> float:
        """Evaluate the cost of a solution after applying a move."""
        if move[0] == 'relocate':
            new_solution = self._apply_relocate(solution, move)
        elif move[0] == 'exchange':
            new_solution = self._apply_exchange(solution, move)
        else:
            return float('inf')

        return compute_solution_cost(new_solution, self.dm)

    def _get_move_key(self, move: Tuple) -> Tuple:
        """Get a hashable key for a move for tabu checking."""
        if move[0] == 'relocate':
            return ('relocate', move[1])  # Just track the customer
        elif move[0] == 'exchange':
            return ('exchange', tuple(sorted([move[1], move[2]])))
        return move

    def search(self, initial_solution: List[List[int]]) -> Tuple[List[List[int]], Dict[str, Any]]:
        """
        Run Tabu Search.

        Args:
            initial_solution: Starting solution

        Returns:
            Tuple of (best_solution, statistics)
        """
        current_solution = [route[:] for route in initial_solution]
        current_cost = compute_solution_cost(current_solution, self.dm)

        best_solution = [route[:] for route in current_solution]
        best_cost = current_cost

        self.stats['initial_cost'] = current_cost
        self.stats['cost_history'] = [current_cost]

        no_improve_count = 0
        iteration = 0

        while iteration < self.max_iterations and no_improve_count < self.max_no_improve:
            iteration += 1
            self.stats['iterations'] = iteration

            # Generate neighborhood moves
            moves = self._get_relocate_moves(current_solution)
            moves.extend(self._get_exchange_moves(current_solution))

            if not moves:
                break

            # Evaluate all moves
            best_move = None
            best_move_cost = float('inf')

            random.shuffle(moves)  # Randomize to break ties

            for move in moves:
                move_key = self._get_move_key(move)
                is_tabu = self._is_tabu(move_key, iteration)

                new_cost = self._evaluate_move(current_solution, move)

                # Aspiration criteria: accept if better than best known
                if is_tabu and self.aspiration:
                    if new_cost < best_cost:
                        is_tabu = False
                        self.stats['aspiration_used'] += 1

                if not is_tabu and new_cost < best_move_cost:
                    best_move = move
                    best_move_cost = new_cost

            # Apply best move
            if best_move is not None:
                if best_move[0] == 'relocate':
                    current_solution = self._apply_relocate(current_solution, best_move)
                elif best_move[0] == 'exchange':
                    current_solution = self._apply_exchange(current_solution, best_move)

                current_cost = best_move_cost

                # Add to tabu list
                move_key = self._get_move_key(best_move)
                self._add_tabu(move_key, iteration)

                # Update best
                if current_cost < best_cost:
                    best_solution = [route[:] for route in current_solution]
                    best_cost = current_cost
                    self.stats['improvements'] += 1
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1

            self.stats['cost_history'].append(current_cost)

            # Diversification: random restart if stuck
            if self.diversification and no_improve_count >= self.max_no_improve // 2:
                if random.random() < 0.3:
                    current_solution = self._diversify(current_solution)
                    current_cost = compute_solution_cost(current_solution, self.dm)
                    self.stats['diversification_count'] += 1
                    no_improve_count = 0

            if self.verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: Current = {current_cost:.2f}, Best = {best_cost:.2f}")

        self.stats['final_cost'] = best_cost
        self.stats['improvement'] = self.stats['initial_cost'] - best_cost

        return best_solution, self.stats

    def _diversify(self, solution: List[List[int]]) -> List[List[int]]:
        """Diversification: make random perturbations."""
        new_solution = [route[:] for route in solution]

        # Randomly select two routes and exchange some customers
        if len(new_solution) >= 2:
            r1_idx, r2_idx = random.sample(range(len(new_solution)), 2)
            route1, route2 = new_solution[r1_idx], new_solution[r2_idx]

            if len(route1) > 3 and len(route2) > 3:
                # Select random customers
                i = random.randint(1, len(route1) - 2)
                j = random.randint(1, len(route2) - 2)

                c1, c2 = route1[i], route2[j]

                # Check capacity before exchange
                d1, d2 = self.demands.get(c1, 0), self.demands.get(c2, 0)
                load1 = get_route_load(route1, self.demands, self.depot) - d1 + d2
                load2 = get_route_load(route2, self.demands, self.depot) - d2 + d1

                if load1 <= self.capacity and load2 <= self.capacity:
                    route1[i] = c2
                    route2[j] = c1

        return new_solution


def tabu_search(instance: Dict[str, Any],
                initial_solution: Optional[List[List[int]]] = None,
                tabu_tenure: int = 10,
                max_iterations: int = 1000,
                max_no_improve: int = 100,
                **kwargs) -> Tuple[List[List[int]], Dict[str, Any]]:
    """
    Run Tabu Search on a CVRP instance.

    Args:
        instance: CVRP instance
        initial_solution: Starting solution (if None, uses nearest neighbor)
        tabu_tenure: Tabu list tenure
        max_iterations: Maximum iterations
        max_no_improve: Stopping criterion
        **kwargs: Additional parameters for TabuSearch

    Returns:
        Tuple of (best_solution, statistics)
    """
    if initial_solution is None:
        from heuristics.greedy import nearest_neighbor
        initial_solution = nearest_neighbor(instance)

    ts = TabuSearch(
        instance,
        tabu_tenure=tabu_tenure,
        max_iterations=max_iterations,
        max_no_improve=max_no_improve,
        **kwargs
    )

    return ts.search(initial_solution)


def run_tabu_experiments(instance: Dict[str, Any],
                         tenure_values: List[int] = [5, 10, 15, 20, 30],
                         num_runs: int = 5) -> Dict[str, Any]:
    """
    Run experiments with different tabu tenures.

    Args:
        instance: CVRP instance
        tenure_values: List of tenure values to test
        num_runs: Number of runs per configuration

    Returns:
        Dictionary with experiment results
    """
    from heuristics.greedy import nearest_neighbor, savings_algorithm
    from common.utils import Timer

    results = {
        'tenure_values': tenure_values,
        'experiments': []
    }

    initial_solutions = {
        'Nearest Neighbor': nearest_neighbor(instance),
        'Savings': savings_algorithm(instance)
    }

    for init_name, init_sol in initial_solutions.items():
        for tenure in tenure_values:
            for run in range(num_runs):
                with Timer() as t:
                    solution, stats = tabu_search(
                        instance,
                        initial_solution=[route[:] for route in init_sol],
                        tabu_tenure=tenure,
                        max_iterations=500,
                        max_no_improve=50
                    )

                cost = compute_solution_cost(solution, instance['distance_matrix'])
                gap = None
                if instance.get('optimal'):
                    gap = ((cost - instance['optimal']) / instance['optimal']) * 100

                results['experiments'].append({
                    'initial': init_name,
                    'tenure': tenure,
                    'run': run + 1,
                    'cost': cost,
                    'gap': gap,
                    'time': t.elapsed,
                    'iterations': stats['iterations'],
                    'improvements': stats['improvements']
                })

    return results


if __name__ == '__main__':
    from common.reader import read_instance
    from common.utils import Timer, print_result, create_result
    from heuristics.greedy import nearest_neighbor

    print("Tabu Search for CVRP")
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
    print(f"\nInitial solution cost: {initial_cost:.2f}")

    # Run Tabu Search with different parameters
    print("\nTesting different tabu tenures:")
    print("-" * 50)

    for tenure in [5, 10, 15, 20]:
        with Timer() as t:
            solution, stats = tabu_search(
                instance,
                initial_solution=[route[:] for route in initial],
                tabu_tenure=tenure,
                max_iterations=500,
                max_no_improve=50,
                verbose=False
            )

        cost = compute_solution_cost(solution, instance['distance_matrix'])
        print(f"Tenure {tenure:>2}: Cost = {cost:>8.2f}, "
              f"Iterations = {stats['iterations']:>4}, "
              f"Improvements = {stats['improvements']:>3}, "
              f"Time = {t.elapsed:.4f}s")

    print("\nBest solution routes:")
    for i, route in enumerate(solution):
        print(f"  Route {i + 1}: {' -> '.join(map(str, route))}")
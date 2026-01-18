"""
Simulated Annealing for CVRP
Team 2 - Soft Computing Contest
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Any, Tuple, Optional
import random
import math
import copy
from common.cost import compute_solution_cost, compute_route_cost
from common.feasibility import is_feasible, get_route_load


class SimulatedAnnealing:
    """
    Simulated Annealing implementation for CVRP.

    Features:
    - Multiple neighborhood operators
    - Various cooling schedules
    - Reheating mechanism
    - Adaptive temperature control
    """

    def __init__(self, instance: Dict[str, Any],
                 initial_temp: float = 1000.0,
                 final_temp: float = 0.1,
                 cooling_rate: float = 0.995,
                 iterations_per_temp: int = 100,
                 cooling_schedule: str = 'geometric',
                 reheat: bool = False,
                 reheat_threshold: int = 50,
                 verbose: bool = False):
        """
        Initialize Simulated Annealing.

        Args:
            instance: CVRP instance
            initial_temp: Starting temperature
            final_temp: Stopping temperature
            cooling_rate: Temperature reduction rate
            iterations_per_temp: Iterations at each temperature
            cooling_schedule: 'geometric', 'linear', 'logarithmic', 'adaptive'
            reheat: Enable reheating mechanism
            reheat_threshold: Iterations without improvement before reheat
            verbose: Print progress
        """
        self.instance = instance
        self.depot = instance['depot']
        self.capacity = instance['capacity']
        self.demands = instance['demands']
        self.dm = instance['distance_matrix']

        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.iterations_per_temp = iterations_per_temp
        self.cooling_schedule = cooling_schedule
        self.reheat = reheat
        self.reheat_threshold = reheat_threshold
        self.verbose = verbose

        # Statistics
        self.stats = {
            'total_iterations': 0,
            'accepted': 0,
            'accepted_worse': 0,
            'improvements': 0,
            'reheats': 0,
            'cost_history': [],
            'temp_history': []
        }

    def _random_relocate(self, solution: List[List[int]]) -> Optional[List[List[int]]]:
        """Generate a neighbor by relocating a random customer."""
        new_solution = [route[:] for route in solution]

        # Find non-empty routes
        valid_routes = [(i, r) for i, r in enumerate(new_solution) if len(r) > 2]
        if not valid_routes:
            return None

        # Select random source route and customer
        r1_idx, route1 = random.choice(valid_routes)
        if len(route1) <= 2:
            return None

        pos1 = random.randint(1, len(route1) - 2)
        customer = route1[pos1]
        customer_demand = self.demands.get(customer, 0)

        # Remove customer from source
        new_route1 = route1[:pos1] + route1[pos1 + 1:]

        # Select destination route
        r2_idx = random.randint(0, len(new_solution) - 1)

        if r1_idx == r2_idx:
            route2 = new_route1
        else:
            route2 = new_solution[r2_idx]
            # Check capacity
            if get_route_load(route2, self.demands, self.depot) + customer_demand > self.capacity:
                return None

        # Insert at random position
        pos2 = random.randint(1, len(route2) - 1)
        new_route2 = route2[:pos2] + [customer] + route2[pos2:]

        # Update solution
        if r1_idx == r2_idx:
            new_solution[r1_idx] = new_route2
        else:
            new_solution[r1_idx] = new_route1
            new_solution[r2_idx] = new_route2

        # Remove empty routes
        new_solution = [r for r in new_solution if len(r) > 2]

        return new_solution

    def _random_exchange(self, solution: List[List[int]]) -> Optional[List[List[int]]]:
        """Generate a neighbor by exchanging two customers."""
        new_solution = [route[:] for route in solution]

        # Find routes with customers
        valid_routes = [(i, r) for i, r in enumerate(new_solution) if len(r) > 2]
        if len(valid_routes) < 1:
            return None

        # Select two routes (can be same)
        r1_idx, route1 = random.choice(valid_routes)
        r2_idx, route2 = random.choice(valid_routes)

        # Select customers
        pos1 = random.randint(1, len(route1) - 2)

        if r1_idx == r2_idx:
            if len(route1) <= 3:
                return None
            pos2 = random.randint(1, len(route1) - 2)
            while pos2 == pos1:
                pos2 = random.randint(1, len(route1) - 2)
        else:
            pos2 = random.randint(1, len(route2) - 2)

        c1 = route1[pos1]
        c2 = route2[pos2]

        # Check capacity
        if r1_idx != r2_idx:
            d1, d2 = self.demands.get(c1, 0), self.demands.get(c2, 0)
            load1 = get_route_load(route1, self.demands, self.depot) - d1 + d2
            load2 = get_route_load(route2, self.demands, self.depot) - d2 + d1

            if load1 > self.capacity or load2 > self.capacity:
                return None

        # Perform exchange
        new_solution[r1_idx][pos1] = c2
        new_solution[r2_idx][pos2] = c1

        return new_solution

    def _random_2opt(self, solution: List[List[int]]) -> Optional[List[List[int]]]:
        """Generate a neighbor using 2-opt on a random route."""
        new_solution = [route[:] for route in solution]

        # Select route with enough customers
        valid_routes = [(i, r) for i, r in enumerate(new_solution) if len(r) > 4]
        if not valid_routes:
            return None

        r_idx, route = random.choice(valid_routes)

        # Select two positions to reverse between
        i = random.randint(1, len(route) - 3)
        j = random.randint(i + 1, len(route) - 2)

        # Reverse segment
        new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
        new_solution[r_idx] = new_route

        return new_solution

    def _random_or_opt(self, solution: List[List[int]]) -> Optional[List[List[int]]]:
        """Generate a neighbor by moving a segment of customers."""
        new_solution = [route[:] for route in solution]

        # Select source route
        valid_routes = [(i, r) for i, r in enumerate(new_solution) if len(r) > 3]
        if not valid_routes:
            return None

        r1_idx, route1 = random.choice(valid_routes)

        # Select segment (1-3 customers)
        seg_size = min(random.randint(1, 3), len(route1) - 2)
        pos1 = random.randint(1, len(route1) - 1 - seg_size)
        segment = route1[pos1:pos1 + seg_size]
        segment_demand = sum(self.demands.get(c, 0) for c in segment)

        # Remove segment
        new_route1 = route1[:pos1] + route1[pos1 + seg_size:]

        # Select destination route
        r2_idx = random.randint(0, len(new_solution) - 1)

        if r1_idx == r2_idx:
            base_route = new_route1
        else:
            base_route = new_solution[r2_idx]
            # Check capacity
            if get_route_load(base_route, self.demands, self.depot) + segment_demand > self.capacity:
                return None

        # Insert at random position
        if len(base_route) < 2:
            return None
        pos2 = random.randint(1, len(base_route) - 1)
        new_route2 = base_route[:pos2] + segment + base_route[pos2:]

        # Update solution
        if r1_idx == r2_idx:
            new_solution[r1_idx] = new_route2
        else:
            new_solution[r1_idx] = new_route1
            new_solution[r2_idx] = new_route2

        # Remove empty routes
        new_solution = [r for r in new_solution if len(r) > 2]

        return new_solution

    def _get_neighbor(self, solution: List[List[int]]) -> Optional[List[List[int]]]:
        """Generate a random neighbor using one of the operators."""
        operators = [
            self._random_relocate,
            self._random_exchange,
            self._random_2opt,
            self._random_or_opt
        ]

        # Try operators until one succeeds
        random.shuffle(operators)
        for op in operators:
            neighbor = op(solution)
            if neighbor is not None and is_feasible(neighbor, self.instance):
                return neighbor

        return None

    def _cool_temperature(self, temp: float, iteration: int, max_iterations: int) -> float:
        """Apply cooling schedule."""
        if self.cooling_schedule == 'geometric':
            return temp * self.cooling_rate
        elif self.cooling_schedule == 'linear':
            return self.initial_temp - (self.initial_temp - self.final_temp) * (iteration / max_iterations)
        elif self.cooling_schedule == 'logarithmic':
            return self.initial_temp / (1 + math.log(1 + iteration))
        elif self.cooling_schedule == 'adaptive':
            # Adaptive: cool faster when accepting, slower when rejecting
            return temp * self.cooling_rate
        else:
            return temp * self.cooling_rate

    def search(self, initial_solution: List[List[int]]) -> Tuple[List[List[int]], Dict[str, Any]]:
        """
        Run Simulated Annealing.

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
        self.stats['temp_history'] = [self.initial_temp]

        temp = self.initial_temp
        no_improve_count = 0
        iteration = 0

        # Estimate max iterations for linear cooling
        max_iterations = int(math.log(self.final_temp / self.initial_temp) /
                             math.log(self.cooling_rate)) * self.iterations_per_temp

        while temp > self.final_temp:
            for _ in range(self.iterations_per_temp):
                iteration += 1
                self.stats['total_iterations'] = iteration

                # Generate neighbor
                neighbor = self._get_neighbor(current_solution)

                if neighbor is None:
                    continue

                neighbor_cost = compute_solution_cost(neighbor, self.dm)
                delta = neighbor_cost - current_cost

                # Accept or reject
                accept = False
                if delta < 0:
                    accept = True
                else:
                    # Accept worse solution with probability exp(-delta/T)
                    if temp > 0:
                        prob = math.exp(-delta / temp)
                        if random.random() < prob:
                            accept = True
                            self.stats['accepted_worse'] += 1

                if accept:
                    current_solution = neighbor
                    current_cost = neighbor_cost
                    self.stats['accepted'] += 1

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
            self.stats['temp_history'].append(temp)

            # Cool down
            temp = self._cool_temperature(temp, iteration, max_iterations)

            # Reheating mechanism
            if self.reheat and no_improve_count > self.reheat_threshold:
                temp = self.initial_temp * 0.5
                self.stats['reheats'] += 1
                no_improve_count = 0

            if self.verbose and iteration % 1000 == 0:
                print(f"Iteration {iteration}: T={temp:.2f}, Current={current_cost:.2f}, Best={best_cost:.2f}")

        self.stats['final_cost'] = best_cost
        self.stats['improvement'] = self.stats['initial_cost'] - best_cost
        self.stats['final_temp'] = temp

        return best_solution, self.stats


def simulated_annealing(instance: Dict[str, Any],
                        initial_solution: Optional[List[List[int]]] = None,
                        initial_temp: float = 1000.0,
                        final_temp: float = 0.1,
                        cooling_rate: float = 0.995,
                        **kwargs) -> Tuple[List[List[int]], Dict[str, Any]]:
    """
    Run Simulated Annealing on a CVRP instance.

    Args:
        instance: CVRP instance
        initial_solution: Starting solution (if None, uses nearest neighbor)
        initial_temp: Starting temperature
        final_temp: Stopping temperature
        cooling_rate: Temperature reduction rate
        **kwargs: Additional parameters for SimulatedAnnealing

    Returns:
        Tuple of (best_solution, statistics)
    """
    if initial_solution is None:
        from heuristics.greedy import nearest_neighbor
        initial_solution = nearest_neighbor(instance)

    sa = SimulatedAnnealing(
        instance,
        initial_temp=initial_temp,
        final_temp=final_temp,
        cooling_rate=cooling_rate,
        **kwargs
    )

    return sa.search(initial_solution)


def estimate_initial_temperature(instance: Dict[str, Any],
                                 initial_solution: List[List[int]],
                                 target_acceptance: float = 0.8,
                                 samples: int = 100) -> float:
    """
    Estimate a good initial temperature based on target acceptance rate.

    Args:
        instance: CVRP instance
        initial_solution: Starting solution
        target_acceptance: Desired initial acceptance rate
        samples: Number of samples for estimation

    Returns:
        Estimated initial temperature
    """
    dm = instance['distance_matrix']

    sa = SimulatedAnnealing(instance)
    current_cost = compute_solution_cost(initial_solution, dm)

    # Sample random moves and compute deltas
    deltas = []
    for _ in range(samples):
        neighbor = sa._get_neighbor(initial_solution)
        if neighbor is not None:
            neighbor_cost = compute_solution_cost(neighbor, dm)
            delta = neighbor_cost - current_cost
            if delta > 0:
                deltas.append(delta)

    if not deltas:
        return 1000.0

    # Compute temperature for target acceptance rate
    avg_delta = sum(deltas) / len(deltas)
    temp = -avg_delta / math.log(target_acceptance)

    return max(temp, 100.0)


def run_sa_experiments(instance: Dict[str, Any],
                       cooling_rates: List[float] = [0.99, 0.995, 0.999],
                       temp_configs: List[Tuple[float, float]] = None,
                       num_runs: int = 5) -> Dict[str, Any]:
    """
    Run experiments with different SA configurations.

    Args:
        instance: CVRP instance
        cooling_rates: List of cooling rates to test
        temp_configs: List of (initial_temp, final_temp) tuples
        num_runs: Number of runs per configuration

    Returns:
        Dictionary with experiment results
    """
    from heuristics.greedy import nearest_neighbor, savings_algorithm
    from common.utils import Timer

    if temp_configs is None:
        temp_configs = [(500.0, 0.1), (1000.0, 0.1), (2000.0, 0.1)]

    results = {
        'cooling_rates': cooling_rates,
        'temp_configs': temp_configs,
        'experiments': []
    }

    initial_solutions = {
        'Nearest Neighbor': nearest_neighbor(instance),
        'Savings': savings_algorithm(instance)
    }

    for init_name, init_sol in initial_solutions.items():
        for init_temp, final_temp in temp_configs:
            for cooling_rate in cooling_rates:
                for run in range(num_runs):
                    with Timer() as t:
                        solution, stats = simulated_annealing(
                            instance,
                            initial_solution=[route[:] for route in init_sol],
                            initial_temp=init_temp,
                            final_temp=final_temp,
                            cooling_rate=cooling_rate,
                            iterations_per_temp=50
                        )

                    cost = compute_solution_cost(solution, instance['distance_matrix'])
                    gap = None
                    if instance.get('optimal'):
                        gap = ((cost - instance['optimal']) / instance['optimal']) * 100

                    results['experiments'].append({
                        'initial': init_name,
                        'init_temp': init_temp,
                        'final_temp': final_temp,
                        'cooling_rate': cooling_rate,
                        'run': run + 1,
                        'cost': cost,
                        'gap': gap,
                        'time': t.elapsed,
                        'iterations': stats['total_iterations'],
                        'accepted': stats['accepted'],
                        'improvements': stats['improvements']
                    })

    return results


if __name__ == '__main__':
    from common.reader import read_instance
    from common.utils import Timer, print_result, create_result
    from heuristics.greedy import nearest_neighbor

    print("Simulated Annealing for CVRP")
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

    # Estimate initial temperature
    est_temp = estimate_initial_temperature(instance, initial)
    print(f"Estimated initial temperature: {est_temp:.2f}")

    # Run SA with different configurations
    print("\nTesting different cooling rates:")
    print("-" * 60)

    for cooling_rate in [0.99, 0.995, 0.999]:
        with Timer() as t:
            solution, stats = simulated_annealing(
                instance,
                initial_solution=[route[:] for route in initial],
                initial_temp=est_temp,
                final_temp=0.1,
                cooling_rate=cooling_rate,
                iterations_per_temp=50,
                verbose=False
            )

        cost = compute_solution_cost(solution, instance['distance_matrix'])
        print(f"Rate {cooling_rate}: Cost = {cost:>8.2f}, "
              f"Iterations = {stats['total_iterations']:>5}, "
              f"Accepted = {stats['accepted']:>4}, "
              f"Improvements = {stats['improvements']:>3}, "
              f"Time = {t.elapsed:.4f}s")

    print("\nTesting different cooling schedules:")
    print("-" * 60)

    for schedule in ['geometric', 'linear', 'logarithmic']:
        with Timer() as t:
            solution, stats = simulated_annealing(
                instance,
                initial_solution=[route[:] for route in initial],
                initial_temp=est_temp,
                final_temp=0.1,
                cooling_rate=0.995,
                iterations_per_temp=50,
                cooling_schedule=schedule,
                verbose=False
            )

        cost = compute_solution_cost(solution, instance['distance_matrix'])
        print(f"{schedule:<12}: Cost = {cost:>8.2f}, "
              f"Iterations = {stats['total_iterations']:>5}, "
              f"Time = {t.elapsed:.4f}s")

    print("\nBest solution routes:")
    for i, route in enumerate(solution):
        print(f"  Route {i + 1}: {' -> '.join(map(str, route))}")
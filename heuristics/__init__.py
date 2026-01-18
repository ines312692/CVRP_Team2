"""
Heuristics package for CVRP
Team 2 - Soft Computing Contest
"""

from .greedy import (
    nearest_neighbor,
    nearest_neighbor_enhanced,
    savings_algorithm,
    savings_algorithm_sequential,
    sequential_insertion,
    sweep_algorithm,
    compare_greedy_heuristics
)

from .local_search import (
    two_opt,
    two_opt_single_route,
    or_opt,
    relocate,
    exchange,
    cross_exchange,
    local_search,
    variable_neighborhood_descent
)

from .tabu_search import (
    TabuSearch,
    tabu_search,
    run_tabu_experiments
)

from .simulated_annealing import (
    SimulatedAnnealing,
    simulated_annealing,
    estimate_initial_temperature,
    run_sa_experiments
)

__all__ = [
    # Greedy
    'nearest_neighbor', 'nearest_neighbor_enhanced',
    'savings_algorithm', 'savings_algorithm_sequential',
    'sequential_insertion', 'sweep_algorithm', 'compare_greedy_heuristics',
    # Local Search
    'two_opt', 'two_opt_single_route', 'or_opt', 'relocate',
    'exchange', 'cross_exchange', 'local_search', 'variable_neighborhood_descent',
    # Tabu Search
    'TabuSearch', 'tabu_search', 'run_tabu_experiments',
    # Simulated Annealing
    'SimulatedAnnealing', 'simulated_annealing',
    'estimate_initial_temperature', 'run_sa_experiments'
]
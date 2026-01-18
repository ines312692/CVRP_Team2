"""
Main Comparison and Benchmarking Script for Team 2
CVRP Soft Computing Contest

This script runs all implemented metaheuristics on given instances
and produces comparison results according to the specified indicators.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import statistics

from common.reader import read_instance, get_customers, get_instance_info
from common.cost import compute_solution_cost, compute_gap, get_cost_breakdown
from common.feasibility import is_feasible, check_feasibility, get_feasibility_report
from common.utils import (
    create_result, save_result, save_all_results, Timer,
    compute_statistics, format_solution, print_result, get_instance_files
)

from heuristics.greedy import (
    nearest_neighbor, nearest_neighbor_enhanced, savings_algorithm,
    savings_algorithm_sequential, sequential_insertion, sweep_algorithm
)
from heuristics.local_search import (
    two_opt, local_search, variable_neighborhood_descent
)
from heuristics.tabu_search import tabu_search
from heuristics.simulated_annealing import simulated_annealing, estimate_initial_temperature


# =============================================================================
# ALGORITHM RUNNERS
# =============================================================================

def run_greedy(instance: Dict[str, Any], method_name: str) -> Dict[str, Any]:
    """Run a greedy heuristic."""
    greedy_methods = {
        'Nearest Neighbor': nearest_neighbor,
        'Nearest Neighbor Enhanced': nearest_neighbor_enhanced,
        'Savings Parallel': lambda i: savings_algorithm(i, parallel=True),
        'Savings Sequential': savings_algorithm_sequential,
        'Sequential Insertion': sequential_insertion,
        'Sweep': sweep_algorithm
    }

    if method_name not in greedy_methods:
        raise ValueError(f"Unknown greedy method: {method_name}")

    with Timer() as t:
        solution = greedy_methods[method_name](instance)

    cost = compute_solution_cost(solution, instance['distance_matrix'])
    feasible = is_feasible(solution, instance)

    return create_result(
        instance_name=instance['name'],
        method=method_name,
        best_cost=cost,
        execution_time=t.elapsed,
        feasible=feasible,
        solution=solution,
        optimal=instance.get('optimal')
    )


def run_local_search(instance: Dict[str, Any],
                     initial_method: str = 'Nearest Neighbor',
                     operators: List[str] = None) -> Dict[str, Any]:
    """Run local search with specified initial solution."""
    # Get initial solution
    if initial_method == 'Nearest Neighbor':
        initial = nearest_neighbor(instance)
    elif initial_method == 'Savings':
        initial = savings_algorithm(instance)
    else:
        initial = nearest_neighbor(instance)

    if operators is None:
        operators = ['2opt', 'relocate', 'exchange', 'oropt']

    with Timer() as t:
        solution, info = local_search(initial, instance, operators=operators)

    cost = compute_solution_cost(solution, instance['distance_matrix'])
    feasible = is_feasible(solution, instance)

    return create_result(
        instance_name=instance['name'],
        method=f'Local Search ({initial_method})',
        best_cost=cost,
        execution_time=t.elapsed,
        feasible=feasible,
        solution=solution,
        optimal=instance.get('optimal'),
        initial_cost=info['initial_cost'],
        iterations=info['iterations'],
        improvements=info['improvements']
    )


def run_tabu_search(instance: Dict[str, Any],
                    initial_method: str = 'Nearest Neighbor',
                    tabu_tenure: int = 10,
                    max_iterations: int = 500,
                    max_no_improve: int = 50) -> Dict[str, Any]:
    """Run Tabu Search with specified parameters."""
    # Get initial solution
    if initial_method == 'Nearest Neighbor':
        initial = nearest_neighbor(instance)
    elif initial_method == 'Savings':
        initial = savings_algorithm(instance)
    else:
        initial = nearest_neighbor(instance)

    with Timer() as t:
        solution, stats = tabu_search(
            instance,
            initial_solution=initial,
            tabu_tenure=tabu_tenure,
            max_iterations=max_iterations,
            max_no_improve=max_no_improve
        )

    cost = compute_solution_cost(solution, instance['distance_matrix'])
    feasible = is_feasible(solution, instance)

    return create_result(
        instance_name=instance['name'],
        method=f'Tabu Search (tenure={tabu_tenure})',
        best_cost=cost,
        execution_time=t.elapsed,
        feasible=feasible,
        solution=solution,
        optimal=instance.get('optimal'),
        initial_cost=stats['initial_cost'],
        iterations=stats['iterations'],
        improvements=stats['improvements'],
        aspiration_used=stats['aspiration_used'],
        tabu_tenure=tabu_tenure,
        initial_method=initial_method
    )


def run_simulated_annealing(instance: Dict[str, Any],
                            initial_method: str = 'Nearest Neighbor',
                            initial_temp: float = 1000.0,
                            final_temp: float = 0.1,
                            cooling_rate: float = 0.995) -> Dict[str, Any]:
    """Run Simulated Annealing with specified parameters."""
    # Get initial solution
    if initial_method == 'Nearest Neighbor':
        initial = nearest_neighbor(instance)
    elif initial_method == 'Savings':
        initial = savings_algorithm(instance)
    else:
        initial = nearest_neighbor(instance)

    # Estimate temperature if using auto
    if initial_temp is None:
        initial_temp = estimate_initial_temperature(instance, initial)

    with Timer() as t:
        solution, stats = simulated_annealing(
            instance,
            initial_solution=initial,
            initial_temp=initial_temp,
            final_temp=final_temp,
            cooling_rate=cooling_rate,
            iterations_per_temp=50
        )

    cost = compute_solution_cost(solution, instance['distance_matrix'])
    feasible = is_feasible(solution, instance)

    return create_result(
        instance_name=instance['name'],
        method=f'Simulated Annealing (α={cooling_rate})',
        best_cost=cost,
        execution_time=t.elapsed,
        feasible=feasible,
        solution=solution,
        optimal=instance.get('optimal'),
        initial_cost=stats['initial_cost'],
        total_iterations=stats['total_iterations'],
        accepted=stats['accepted'],
        improvements=stats['improvements'],
        cooling_rate=cooling_rate,
        initial_temp=initial_temp,
        initial_method=initial_method
    )


# =============================================================================
# BENCHMARK SUITE
# =============================================================================

def run_full_benchmark(instance: Dict[str, Any],
                       verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Run all algorithms on a single instance.

    Args:
        instance: CVRP instance
        verbose: Print progress

    Returns:
        List of result dictionaries
    """
    results = []

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {instance['name']}")
        print(f"{'=' * 60}")

    # 1. Greedy Heuristics
    greedy_methods = [
        'Nearest Neighbor', 'Nearest Neighbor Enhanced',
        'Savings Parallel', 'Savings Sequential',
        'Sequential Insertion', 'Sweep'
    ]

    if verbose:
        print("\n--- Greedy Heuristics ---")

    for method in greedy_methods:
        result = run_greedy(instance, method)
        results.append(result)
        if verbose:
            print(f"{method:<30}: Cost={result['best_cost']:>10.2f}, "
                  f"Time={result['execution_time']:.4f}s")

    # 2. Local Search
    if verbose:
        print("\n--- Local Search ---")

    for init_method in ['Nearest Neighbor', 'Savings']:
        result = run_local_search(instance, initial_method=init_method)
        results.append(result)
        if verbose:
            print(f"LS ({init_method}): Cost={result['best_cost']:>10.2f}, "
                  f"Time={result['execution_time']:.4f}s")

    # 3. Tabu Search
    if verbose:
        print("\n--- Tabu Search ---")

    for tenure in [5, 10, 15]:
        for init_method in ['Nearest Neighbor', 'Savings']:
            result = run_tabu_search(
                instance,
                initial_method=init_method,
                tabu_tenure=tenure
            )
            results.append(result)
            if verbose:
                print(f"TS (tenure={tenure}, {init_method}): "
                      f"Cost={result['best_cost']:>10.2f}, "
                      f"Time={result['execution_time']:.4f}s")

    # 4. Simulated Annealing
    if verbose:
        print("\n--- Simulated Annealing ---")

    for cooling_rate in [0.99, 0.995, 0.999]:
        for init_method in ['Nearest Neighbor', 'Savings']:
            result = run_simulated_annealing(
                instance,
                initial_method=init_method,
                cooling_rate=cooling_rate
            )
            results.append(result)
            if verbose:
                print(f"SA (α={cooling_rate}, {init_method}): "
                      f"Cost={result['best_cost']:>10.2f}, "
                      f"Time={result['execution_time']:.4f}s")

    return results


def run_instance_benchmark(filepath: str,
                           output_dir: str = 'results',
                           verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Run benchmark on a single instance file.

    Args:
        filepath: Path to instance file
        output_dir: Directory for results
        verbose: Print progress

    Returns:
        List of results
    """
    instance = read_instance(filepath)
    results = run_full_benchmark(instance, verbose=verbose)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{instance['name']}_results.json")
    save_all_results(results, output_file)

    return results


def run_directory_benchmark(directory: str,
                            output_dir: str = 'results',
                            verbose: bool = True) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run benchmark on all instances in a directory.

    Args:
        directory: Directory containing instance files
        output_dir: Directory for results
        verbose: Print progress

    Returns:
        Dictionary mapping instance name to results
    """
    all_results = {}
    instance_files = get_instance_files(directory)

    if not instance_files:
        print(f"No .vrp files found in {directory}")
        return all_results

    print(f"\nFound {len(instance_files)} instances")

    for filepath in instance_files:
        try:
            results = run_instance_benchmark(filepath, output_dir, verbose)
            instance_name = os.path.basename(filepath).replace('.vrp', '')
            all_results[instance_name] = results
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    return all_results


# =============================================================================
# COMPARISON AND ANALYSIS
# =============================================================================

def compare_methods(all_results: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Compare all methods across instances and print summary.

    Args:
        all_results: Dictionary mapping instance name to results list
    """
    # Aggregate by method
    method_results = {}

    for instance_name, results in all_results.items():
        for result in results:
            method = result['method']
            if method not in method_results:
                method_results[method] = []
            method_results[method].append(result)

    print("\n" + "=" * 80)
    print("METHOD COMPARISON SUMMARY")
    print("=" * 80)

    # Print header
    print(f"\n{'Method':<40} {'Avg Gap':>10} {'Std Gap':>10} {'Avg Time':>10} {'Feasible':>10}")
    print("-" * 80)

    # Sort by average gap
    method_stats = []
    for method, results in method_results.items():
        stats = compute_statistics(results)
        stats['method'] = method
        method_stats.append(stats)

    method_stats.sort(key=lambda x: x.get('avg_gap', float('inf')))

    for stats in method_stats:
        avg_gap = stats.get('avg_gap', 'N/A')
        std_gap = stats.get('std_gap', 'N/A')
        avg_time = stats.get('avg_time', 0)
        feasible = stats.get('feasibility_rate', 0)

        if isinstance(avg_gap, (int, float)):
            avg_gap_str = f"{avg_gap:.2f}%"
        else:
            avg_gap_str = str(avg_gap)

        if isinstance(std_gap, (int, float)):
            std_gap_str = f"{std_gap:.2f}%"
        else:
            std_gap_str = str(std_gap)

        print(f"{stats['method']:<40} {avg_gap_str:>10} {std_gap_str:>10} "
              f"{avg_time:>9.4f}s {feasible:>9.1f}%")


def generate_detailed_report(all_results: Dict[str, List[Dict[str, Any]]],
                             output_file: str = 'detailed_report.json') -> None:
    """
    Generate a detailed comparison report.

    Args:
        all_results: Dictionary mapping instance name to results list
        output_file: Output file path
    """
    report = {
        'generated_at': datetime.now().isoformat(),
        'num_instances': len(all_results),
        'instances': {},
        'method_summary': {}
    }

    # Method aggregation
    method_results = {}

    for instance_name, results in all_results.items():
        # Best result for this instance
        best_result = min(results, key=lambda x: x['best_cost'] if x['feasible'] else float('inf'))

        report['instances'][instance_name] = {
            'best_method': best_result['method'],
            'best_cost': best_result['best_cost'],
            'optimal': best_result.get('optimal'),
            'gap': best_result.get('gap'),
            'num_methods_tested': len(results)
        }

        for result in results:
            method = result['method']
            if method not in method_results:
                method_results[method] = []
            method_results[method].append(result)

    # Method summary
    for method, results in method_results.items():
        stats = compute_statistics(results)
        report['method_summary'][method] = stats

    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nDetailed report saved to {output_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='CVRP Benchmark - Team 2')
    parser.add_argument('input', nargs='?', default=None,
                        help='Instance file or directory')
    parser.add_argument('-o', '--output', default='results',
                        help='Output directory for results')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Quiet mode (less output)')
    parser.add_argument('--demo', action='store_true',
                        help='Run demo with test instance')

    args = parser.parse_args()

    if args.demo or args.input is None:
        print("Running demo with test instance...")

        # Create test instance
        instance = {
            'name': 'demo-test',
            'depot': 1,
            'capacity': 100,
            'demands': {
                1: 0, 2: 20, 3: 30, 4: 40, 5: 25,
                6: 35, 7: 15, 8: 30, 9: 20, 10: 25
            },
            'coordinates': {
                1: (50, 50), 2: (20, 30), 3: (40, 80), 4: (70, 60),
                5: (30, 50), 6: (60, 30), 7: (80, 70), 8: (25, 65),
                9: (55, 25), 10: (75, 45)
            },
            'optimal': 250.0  # Approximate
        }

        # Compute distance matrix
        instance['distance_matrix'] = {}
        for i in instance['coordinates']:
            instance['distance_matrix'][i] = {}
            for j in instance['coordinates']:
                xi, yi = instance['coordinates'][i]
                xj, yj = instance['coordinates'][j]
                instance['distance_matrix'][i][j] = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5

        # Run benchmark
        results = run_full_benchmark(instance, verbose=not args.quiet)

        # Find best
        best = min(results, key=lambda x: x['best_cost'] if x['feasible'] else float('inf'))
        print(f"\n{'=' * 60}")
        print(f"BEST RESULT: {best['method']}")
        print(f"Cost: {best['best_cost']:.2f}")
        if best.get('gap'):
            print(f"Gap: {best['gap']:.2f}%")
        print(f"Time: {best['execution_time']:.4f}s")
        print(f"{'=' * 60}")

        # Save results
        os.makedirs(args.output, exist_ok=True)
        save_all_results(results, os.path.join(args.output, 'demo_results.json'))

    elif os.path.isdir(args.input):
        all_results = run_directory_benchmark(
            args.input,
            args.output,
            verbose=not args.quiet
        )
        compare_methods(all_results)
        generate_detailed_report(
            all_results,
            os.path.join(args.output, 'detailed_report.json')
        )

    elif os.path.isfile(args.input):
        results = run_instance_benchmark(
            args.input,
            args.output,
            verbose=not args.quiet
        )
        # Find and print best result
        best = min(results, key=lambda x: x['best_cost'] if x['feasible'] else float('inf'))
        print(f"\nBest: {best['method']} with cost {best['best_cost']:.2f}")

    else:
        print(f"Error: {args.input} not found")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
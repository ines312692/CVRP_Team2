"""
CVRP Utilities Module
Result saving, statistics computation, and helper functions
Team 1 - Soft Computing Contest
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import statistics


def save_result(result: Dict[str, Any], filepath: str) -> None:
    """
    Save a result to a JSON file.

    Args:
        result: Result dictionary
        filepath: Output file path
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)


def create_result(instance_name: str, method: str, best_cost: float,
                  execution_time: float, feasible: bool,
                  solution: Optional[List[List[int]]] = None,
                  optimal: Optional[float] = None,
                  **kwargs) -> Dict[str, Any]:
    """
    Create a standardized result dictionary.

    Args:
        instance_name: Name of the instance
        method: Algorithm/method used
        best_cost: Best solution cost found
        execution_time: Time in seconds
        feasible: Whether solution is feasible
        solution: Optional solution representation
        optimal: Optional known optimal value
        **kwargs: Additional fields

    Returns:
        Standardized result dictionary
    """
    result = {
        'instance': instance_name,
        'method': method,
        'best_cost': round(best_cost, 2),
        'execution_time': round(execution_time, 4),
        'feasible': feasible,
        'timestamp': datetime.now().isoformat()
    }

    if optimal is not None:
        result['optimal'] = optimal
        result['gap'] = round(((best_cost - optimal) / optimal) * 100, 2)

    if solution is not None:
        result['solution'] = solution
        result['num_routes'] = len(solution)

    # Add any additional fields
    result.update(kwargs)

    return result


def save_all_results(results: List[Dict[str, Any]], filepath: str) -> None:
    """
    Save multiple results to a JSON file.

    Args:
        results: List of result dictionaries
        filepath: Output file path
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(filepath: str) -> List[Dict[str, Any]]:
    """
    Load results from a JSON file.

    Args:
        filepath: Path to results file

    Returns:
        List of result dictionaries
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    return [data]


def compute_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute statistics over a set of results.

    Args:
        results: List of result dictionaries

    Returns:
        Statistics dictionary
    """
    if not results:
        return {}

    gaps = [r.get('gap') for r in results if r.get('gap') is not None]
    times = [r['execution_time'] for r in results]
    costs = [r['best_cost'] for r in results]
    feasible_count = sum(1 for r in results if r.get('feasible', True))

    stats = {
        'num_instances': len(results),
        'num_feasible': feasible_count,
        'feasibility_rate': round(feasible_count / len(results) * 100, 2),
        'avg_time': round(statistics.mean(times), 4),
        'total_time': round(sum(times), 4),
        'avg_cost': round(statistics.mean(costs), 2),
    }

    if gaps:
        stats['avg_gap'] = round(statistics.mean(gaps), 2)
        stats['min_gap'] = round(min(gaps), 2)
        stats['max_gap'] = round(max(gaps), 2)
        if len(gaps) > 1:
            stats['std_gap'] = round(statistics.stdev(gaps), 2)

    return stats


def format_solution(solution: List[List[int]]) -> str:
    """
    Format a solution for display.

    Args:
        solution: List of routes

    Returns:
        Formatted string
    """
    lines = []
    for i, route in enumerate(solution):
        route_str = ' -> '.join(map(str, route))
        lines.append(f"Route {i + 1}: {route_str}")
    return '\n'.join(lines)


def print_result(result: Dict[str, Any]) -> None:
    """
    Print a result in a formatted way.

    Args:
        result: Result dictionary
    """
    print(f"\n{'=' * 50}")
    print(f"Instance: {result['instance']}")
    print(f"Method: {result['method']}")
    print(f"Best Cost: {result['best_cost']}")
    if 'gap' in result:
        print(f"Gap: {result['gap']}%")
    print(f"Execution Time: {result['execution_time']}s")
    print(f"Feasible: {result['feasible']}")
    if 'num_routes' in result:
        print(f"Number of Routes: {result['num_routes']}")
    print(f"{'=' * 50}")


class Timer:
    """Context manager for timing code execution."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time


def get_instance_files(directory: str, extension: str = '.vrp') -> List[str]:
    """
    Get all instance files from a directory.

    Args:
        directory: Path to directory
        extension: File extension to look for

    Returns:
        List of file paths
    """
    if not os.path.isdir(directory):
        return []

    files = []
    for f in sorted(os.listdir(directory)):
        if f.endswith(extension):
            files.append(os.path.join(directory, f))

    return files


def generate_comparison_table(all_results: Dict[str, List[Dict[str, Any]]]) -> str:
    """
    Generate a comparison table for different methods.

    Args:
        all_results: Dict mapping method name -> list of results

    Returns:
        Formatted comparison table
    """
    methods = list(all_results.keys())
    if not methods:
        return "No results to compare"

    # Get all instances
    instances = set()
    for results in all_results.values():
        for r in results:
            instances.add(r['instance'])
    instances = sorted(instances)

    # Build table
    header = f"{'Instance':<20}" + ''.join(f"{m:<15}" for m in methods)
    separator = '-' * len(header)

    lines = [header, separator]

    for inst in instances:
        line = f"{inst:<20}"
        for method in methods:
            result = next((r for r in all_results[method] if r['instance'] == inst), None)
            if result:
                if 'gap' in result:
                    line += f"{result['gap']:>6.2f}%{'':<7}"
                else:
                    line += f"{result['best_cost']:>12.2f}   "
            else:
                line += f"{'N/A':<15}"
        lines.append(line)

    # Add statistics row
    lines.append(separator)
    stat_line = f"{'Avg Gap:':<20}"
    for method in methods:
        stats = compute_statistics(all_results[method])
        if 'avg_gap' in stats:
            stat_line += f"{stats['avg_gap']:>6.2f}%{'':<7}"
        else:
            stat_line += f"{'N/A':<15}"
    lines.append(stat_line)

    return '\n'.join(lines)


if __name__ == '__main__':
    # Test
    result = create_result(
        instance_name='A-n32-k5',
        method='Test',
        best_cost=850.0,
        execution_time=1.5,
        feasible=True,
        optimal=784.0
    )
    print_result(result)
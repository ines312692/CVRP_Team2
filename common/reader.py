"""
CVRP Instance Reader Module
Reads Augerat benchmark instances (.vrp format)
Team 1 - Soft Computing Contest
"""

import re
import math
from typing import Dict, List, Tuple, Any


def read_instance(filepath: str) -> Dict[str, Any]:
    """
    Read a CVRP instance file in standard .vrp format.

    Args:
        filepath: Path to the .vrp instance file

    Returns:
        Dictionary containing:
        - name: Instance name
        - dimension: Number of nodes (including depot)
        - capacity: Vehicle capacity
        - depot: Depot node index (usually 1)
        - coordinates: Dict mapping node -> (x, y)
        - demands: Dict mapping node -> demand
        - distance_matrix: 2D dict for distances
        - optimal: Known optimal value (if available)
    """
    instance = {
        'name': '',
        'dimension': 0,
        'capacity': 0,
        'depot': 1,
        'coordinates': {},
        'demands': {},
        'distance_matrix': {},
        'optimal': None
    }

    with open(filepath, 'r') as f:
        lines = f.readlines()

    section = None

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # Parse header information
        if line.startswith('NAME'):
            instance['name'] = line.split(':')[-1].strip()
        elif line.startswith('DIMENSION'):
            instance['dimension'] = int(line.split(':')[-1].strip())
        elif line.startswith('CAPACITY'):
            instance['capacity'] = int(line.split(':')[-1].strip())
        elif line.startswith('COMMENT'):
            # Try to extract optimal value from comment
            comment = line.split(':')[-1].strip()
            opt_match = re.search(r'Optimal[:\s]+(\d+\.?\d*)', comment, re.IGNORECASE)
            if opt_match:
                instance['optimal'] = float(opt_match.group(1))
        elif line.startswith('NODE_COORD_SECTION'):
            section = 'coords'
        elif line.startswith('DEMAND_SECTION'):
            section = 'demand'
        elif line.startswith('DEPOT_SECTION'):
            section = 'depot'
        elif line.startswith('EOF') or line.startswith('END'):
            break
        elif section == 'coords':
            parts = line.split()
            if len(parts) >= 3:
                node = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                instance['coordinates'][node] = (x, y)
        elif section == 'demand':
            parts = line.split()
            if len(parts) >= 2:
                node = int(parts[0])
                demand = int(parts[1])
                instance['demands'][node] = demand
        elif section == 'depot':
            if line != '-1':
                try:
                    instance['depot'] = int(line)
                except ValueError:
                    pass

    # Compute distance matrix
    instance['distance_matrix'] = compute_distance_matrix(instance['coordinates'])

    return instance


def compute_distance_matrix(coordinates: Dict[int, Tuple[float, float]]) -> Dict[int, Dict[int, float]]:
    """
    Compute Euclidean distance matrix from coordinates.

    Args:
        coordinates: Dict mapping node -> (x, y)

    Returns:
        2D dict where distance_matrix[i][j] = distance from i to j
    """
    distance_matrix = {}
    nodes = list(coordinates.keys())

    for i in nodes:
        distance_matrix[i] = {}
        for j in nodes:
            if i == j:
                distance_matrix[i][j] = 0.0
            else:
                xi, yi = coordinates[i]
                xj, yj = coordinates[j]
                distance_matrix[i][j] = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

    return distance_matrix


def get_customers(instance: Dict[str, Any]) -> List[int]:
    """
    Get list of customer nodes (excluding depot).

    Args:
        instance: Instance dictionary

    Returns:
        List of customer node indices
    """
    depot = instance['depot']
    return [node for node in instance['coordinates'].keys() if node != depot]


def get_instance_info(instance: Dict[str, Any]) -> str:
    """
    Get a formatted string with instance information.

    Args:
        instance: Instance dictionary

    Returns:
        Formatted info string
    """
    info = f"""
Instance: {instance['name']}
Dimension: {instance['dimension']} nodes
Capacity: {instance['capacity']}
Depot: Node {instance['depot']}
Optimal: {instance['optimal'] if instance['optimal'] else 'Unknown'}
Total demand: {sum(instance['demands'].values())}
"""
    return info


if __name__ == '__main__':
    # Test with a sample instance path
    import sys

    if len(sys.argv) > 1:
        inst = read_instance(sys.argv[1])
        print(get_instance_info(inst))
"""
Download Augerat CVRP benchmark instances
Team 2 - Soft Computing Contest
"""

import os
import urllib.request
import sys

# Augerat Set A and B instances from CVRPLIB
INSTANCES = {
    'A': [
        'A-n32-k5', 'A-n33-k5', 'A-n33-k6', 'A-n34-k5', 'A-n36-k5',
        'A-n37-k5', 'A-n37-k6', 'A-n38-k5', 'A-n39-k5', 'A-n39-k6',
        'A-n44-k6', 'A-n45-k6', 'A-n45-k7', 'A-n46-k7', 'A-n48-k7',
        'A-n53-k7', 'A-n54-k7', 'A-n55-k9', 'A-n60-k9', 'A-n61-k9',
        'A-n62-k8', 'A-n63-k9', 'A-n63-k10', 'A-n64-k9', 'A-n65-k9',
        'A-n69-k9', 'A-n80-k10'
    ],
    'B': [
        'B-n31-k5', 'B-n34-k5', 'B-n35-k5', 'B-n38-k6', 'B-n39-k5',
        'B-n41-k6', 'B-n43-k6', 'B-n44-k7', 'B-n45-k5', 'B-n45-k6',
        'B-n50-k7', 'B-n50-k8', 'B-n51-k7', 'B-n52-k7', 'B-n56-k7',
        'B-n57-k7', 'B-n57-k9', 'B-n63-k10', 'B-n64-k9', 'B-n66-k9',
        'B-n67-k10', 'B-n68-k9', 'B-n78-k10'
    ]
}

# Known optimal solutions (from CVRPLIB)
OPTIMAL_VALUES = {
    'A-n32-k5': 784,
    'A-n33-k5': 661,
    'A-n33-k6': 742,
    'A-n34-k5': 778,
    'A-n36-k5': 799,
    'A-n37-k5': 669,
    'A-n37-k6': 949,
    'A-n38-k5': 730,
    'A-n39-k5': 822,
    'A-n39-k6': 831,
    'A-n44-k6': 937,
    'A-n45-k6': 944,
    'A-n45-k7': 1146,
    'A-n46-k7': 914,
    'A-n48-k7': 1073,
    'A-n53-k7': 1010,
    'A-n54-k7': 1167,
    'A-n55-k9': 1073,
    'A-n60-k9': 1354,
    'A-n61-k9': 1034,
    'A-n62-k8': 1288,
    'A-n63-k9': 1616,
    'A-n63-k10': 1314,
    'A-n64-k9': 1401,
    'A-n65-k9': 1174,
    'A-n69-k9': 1159,
    'A-n80-k10': 1763,
    'B-n31-k5': 672,
    'B-n34-k5': 788,
    'B-n35-k5': 955,
    'B-n38-k6': 805,
    'B-n39-k5': 549,
    'B-n41-k6': 829,
    'B-n43-k6': 742,
    'B-n44-k7': 909,
    'B-n45-k5': 751,
    'B-n45-k6': 678,
    'B-n50-k7': 741,
    'B-n50-k8': 1312,
    'B-n51-k7': 1032,
    'B-n52-k7': 747,
    'B-n56-k7': 707,
    'B-n57-k7': 1153,
    'B-n57-k9': 1598,
    'B-n63-k10': 1496,
    'B-n64-k9': 861,
    'B-n66-k9': 1316,
    'B-n67-k10': 1032,
    'B-n68-k9': 1272,
    'B-n78-k10': 1221
}

BASE_URL = "http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/"


def download_instance(instance_name: str, output_dir: str) -> bool:
    """
    Download a single instance.

    Args:
        instance_name: Name of the instance (e.g., 'A-n32-k5')
        output_dir: Directory to save the instance

    Returns:
        True if successful, False otherwise
    """
    url = f"{BASE_URL}{instance_name}.vrp"
    output_path = os.path.join(output_dir, f"{instance_name}.vrp")

    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"  ✓ Downloaded {instance_name}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to download {instance_name}: {e}")
        return False


def download_all(output_dir: str = 'instances', sets: list = None) -> dict:
    """
    Download all instances from specified sets.

    Args:
        output_dir: Directory to save instances
        sets: List of sets to download ('A', 'B') or None for all

    Returns:
        Dictionary with download statistics
    """
    if sets is None:
        sets = ['A', 'B']

    os.makedirs(output_dir, exist_ok=True)

    stats = {'success': 0, 'failed': 0}

    for set_name in sets:
        if set_name not in INSTANCES:
            print(f"Unknown set: {set_name}")
            continue

        print(f"\nDownloading Set {set_name}...")

        for instance_name in INSTANCES[set_name]:
            if download_instance(instance_name, output_dir):
                stats['success'] += 1
            else:
                stats['failed'] += 1

    print(f"\nDownload complete: {stats['success']} succeeded, {stats['failed']} failed")
    return stats


def create_sample_instance(output_dir: str = 'instances') -> str:
    """
    Create a sample instance file for testing.

    Args:
        output_dir: Directory to save the instance

    Returns:
        Path to created instance
    """
    os.makedirs(output_dir, exist_ok=True)

    instance_content = """NAME : sample-n10-k2
COMMENT : Sample instance for testing (Optimal: ~250)
TYPE : CVRP
DIMENSION : 10
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 100
NODE_COORD_SECTION
1 50 50
2 20 30
3 40 80
4 70 60
5 30 50
6 60 30
7 80 70
8 25 65
9 55 25
10 75 45
DEMAND_SECTION
1 0
2 20
3 30
4 40
5 25
6 35
7 15
8 30
9 20
10 25
DEPOT_SECTION
1
-1
EOF
"""

    output_path = os.path.join(output_dir, 'sample-n10-k2.vrp')
    with open(output_path, 'w') as f:
        f.write(instance_content)

    print(f"Created sample instance: {output_path}")
    return output_path


def get_optimal(instance_name: str) -> int:
    """Get the known optimal value for an instance."""
    return OPTIMAL_VALUES.get(instance_name)


def list_instances() -> None:
    """List all available instances."""
    print("\nAvailable Augerat instances:")
    print("\nSet A:")
    for i, name in enumerate(INSTANCES['A']):
        opt = OPTIMAL_VALUES.get(name, 'Unknown')
        print(f"  {name:<15} (Optimal: {opt})")

    print("\nSet B:")
    for i, name in enumerate(INSTANCES['B']):
        opt = OPTIMAL_VALUES.get(name, 'Unknown')
        print(f"  {name:<15} (Optimal: {opt})")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Download CVRP instances')
    parser.add_argument('-o', '--output', default='instances',
                        help='Output directory')
    parser.add_argument('-s', '--sets', nargs='+', default=['A', 'B'],
                        choices=['A', 'B'],
                        help='Instance sets to download')
    parser.add_argument('--sample', action='store_true',
                        help='Create sample instance only')
    parser.add_argument('--list', action='store_true',
                        help='List available instances')

    args = parser.parse_args()

    if args.list:
        list_instances()
    elif args.sample:
        create_sample_instance(args.output)
    else:
        download_all(args.output, args.sets)
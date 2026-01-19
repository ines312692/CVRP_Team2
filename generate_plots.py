import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results(filename):
    """Charge les résultats JSON"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_cost_comparison(results, instance_name, output_dir='plots'):
    """Graphique comparatif des coûts par méthode"""
    Path(output_dir).mkdir(exist_ok=True)

    # Extraire les données
    methods = []
    costs = []
    optimal = results[0]['optimal']

    # Grouper par famille de méthodes
    greedy = []
    local_search = []
    tabu = []
    sa = []

    for r in results:
        method = r['method']
        cost = r['best_cost']

        if 'Tabu' not in method and 'Annealing' not in method and 'Local Search' not in method:
            greedy.append((method, cost))
        elif 'Local Search' in method:
            local_search.append((method, cost))
        elif 'Tabu' in method:
            tabu.append((method, cost))
        elif 'Annealing' in method:
            sa.append((method, cost))

    # Créer le graphique
    fig, ax = plt.subplots(figsize=(14, 8))

    all_methods = greedy + local_search + tabu + sa
    methods = [m[0] for m in all_methods]
    costs = [m[1] for m in all_methods]

    # Définir les couleurs par famille
    colors = []
    for method, _ in all_methods:
        if any(x in method for x in ['Nearest', 'Savings', 'Insertion', 'Sweep']):
            colors.append('#3498db')  # Bleu pour greedy
        elif 'Local Search' in method:
            colors.append('#2ecc71')  # Vert pour LS
        elif 'Tabu' in method:
            colors.append('#e74c3c')  # Rouge pour Tabu
        elif 'Annealing' in method:
            colors.append('#f39c12')  # Orange pour SA

    # Barres
    x = np.arange(len(methods))
    bars = ax.bar(x, costs, color=colors, alpha=0.8, edgecolor='black')

    # Ligne optimale
    ax.axhline(y=optimal, color='green', linestyle='--', linewidth=2, label=f'Optimal = {optimal}')

    # Configuration
    ax.set_xlabel('Méthodes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coût', fontsize=12, fontweight='bold')
    ax.set_title(f'Comparaison des Coûts - {instance_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.0f}',
                ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{instance_name}_cost_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique coûts sauvegardé: {instance_name}_cost_comparison.png")


def plot_time_comparison(results, instance_name, output_dir='plots'):
    """Graphique comparatif des temps d'exécution"""
    Path(output_dir).mkdir(exist_ok=True)

    methods = [r['method'] for r in results]
    times = [r['execution_time'] for r in results]

    fig, ax = plt.subplots(figsize=(14, 8))

    # Couleurs par famille
    colors = []
    for method in methods:
        if any(x in method for x in ['Nearest', 'Savings', 'Insertion', 'Sweep']):
            colors.append('#3498db')
        elif 'Local Search' in method:
            colors.append('#2ecc71')
        elif 'Tabu' in method:
            colors.append('#e74c3c')
        elif 'Annealing' in method:
            colors.append('#f39c12')

    x = np.arange(len(methods))
    bars = ax.bar(x, times, color=colors, alpha=0.8, edgecolor='black')

    ax.set_xlabel('Méthodes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Temps (secondes)', fontsize=12, fontweight='bold')
    ax.set_title(f'Comparaison des Temps d\'Exécution - {instance_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    ax.set_yscale('log')  # Échelle logarithmique
    ax.grid(axis='y', alpha=0.3)

    # Ajouter les valeurs
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=6, rotation=90)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{instance_name}_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique temps sauvegardé: {instance_name}_time_comparison.png")


def plot_gap_comparison(results, instance_name, output_dir='plots'):
    """Graphique comparatif des gaps"""
    Path(output_dir).mkdir(exist_ok=True)

    methods = [r['method'] for r in results]
    gaps = [r['gap'] for r in results]

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = []
    for method in methods:
        if any(x in method for x in ['Nearest', 'Savings', 'Insertion', 'Sweep']):
            colors.append('#3498db')
        elif 'Local Search' in method:
            colors.append('#2ecc71')
        elif 'Tabu' in method:
            colors.append('#e74c3c')
        elif 'Annealing' in method:
            colors.append('#f39c12')

    x = np.arange(len(methods))
    bars = ax.bar(x, gaps, color=colors, alpha=0.8, edgecolor='black')

    ax.set_xlabel('Méthodes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gap (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Comparaison des Gaps - {instance_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # Ligne de référence à 5%
    ax.axhline(y=5, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Gap 5%')
    ax.axhline(y=10, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Gap 10%')

    # Ajouter les valeurs
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=7)

    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{instance_name}_gap_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique gaps sauvegardé: {instance_name}_gap_comparison.png")


def plot_cost_vs_time(results, instance_name, output_dir='plots'):
    """Graphique coût vs temps (scatter plot)"""
    Path(output_dir).mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Grouper par famille
    families = {
        'Greedy': {'color': '#3498db', 'marker': 'o', 'data': []},
        'Local Search': {'color': '#2ecc71', 'marker': 's', 'data': []},
        'Tabu Search': {'color': '#e74c3c', 'marker': '^', 'data': []},
        'Simulated Annealing': {'color': '#f39c12', 'marker': 'D', 'data': []}
    }

    optimal = results[0]['optimal']

    for r in results:
        method = r['method']
        cost = r['best_cost']
        time = r['execution_time']

        if any(x in method for x in ['Nearest', 'Savings', 'Insertion', 'Sweep']):
            families['Greedy']['data'].append((time, cost, method))
        elif 'Local Search' in method:
            families['Local Search']['data'].append((time, cost, method))
        elif 'Tabu' in method:
            families['Tabu Search']['data'].append((time, cost, method))
        elif 'Annealing' in method:
            families['Simulated Annealing']['data'].append((time, cost, method))

    # Tracer chaque famille
    for family, props in families.items():
        if props['data']:
            times = [d[0] for d in props['data']]
            costs = [d[1] for d in props['data']]
            ax.scatter(times, costs, c=props['color'], marker=props['marker'],
                       s=100, alpha=0.7, edgecolors='black', linewidth=1.5, label=family)

    # Ligne optimale
    ax.axhline(y=optimal, color='green', linestyle='--', linewidth=2, label=f'Optimal = {optimal}')

    ax.set_xlabel('Temps d\'exécution (secondes)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coût', fontsize=12, fontweight='bold')
    ax.set_title(f'Coût vs Temps d\'Exécution - {instance_name}', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{instance_name}_cost_vs_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique coût vs temps sauvegardé: {instance_name}_cost_vs_time.png")


def plot_improvements_comparison(results, instance_name, output_dir='plots'):
    """Graphique des améliorations pour les métaheuristiques"""
    Path(output_dir).mkdir(exist_ok=True)

    # Filtrer les méthodes avec améliorations
    data = []
    for r in results:
        if 'improvements' in r:
            data.append({
                'method': r['method'],
                'initial': r['initial_cost'],
                'final': r['best_cost'],
                'improvements': r['improvements']
            })

    if not data:
        print("⚠ Pas de données d'amélioration disponibles")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    methods = [d['method'] for d in data]
    initial = [d['initial'] for d in data]
    final = [d['final'] for d in data]
    improvements = [d['improvements'] for d in data]

    x = np.arange(len(methods))
    width = 0.35

    # Graphique 1: Coût initial vs final
    bars1 = ax1.bar(x - width / 2, initial, width, label='Initial', color='#e74c3c', alpha=0.7)
    bars2 = ax1.bar(x + width / 2, final, width, label='Final', color='#2ecc71', alpha=0.7)

    ax1.set_xlabel('Méthodes', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Coût', fontsize=11, fontweight='bold')
    ax1.set_title('Évolution du Coût: Initial → Final', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Graphique 2: Nombre d'améliorations
    colors = ['#3498db' if imp > 0 else '#95a5a6' for imp in improvements]
    bars = ax2.bar(x, improvements, color=colors, alpha=0.8, edgecolor='black')

    ax2.set_xlabel('Méthodes', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Nombre d\'améliorations', fontsize=11, fontweight='bold')
    ax2.set_title('Nombre d\'Améliorations par Méthode', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)

    # Ajouter les valeurs
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{instance_name}_improvements.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique améliorations sauvegardé: {instance_name}_improvements.png")


def plot_pareto_front(results, instance_name, output_dir='plots'):
    """Graphique du front de Pareto (qualité vs temps)"""
    Path(output_dir).mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Grouper par famille
    families = {
        'Greedy': {'color': '#3498db', 'marker': 'o'},
        'Local Search': {'color': '#2ecc71', 'marker': 's'},
        'Tabu Search': {'color': '#e74c3c', 'marker': '^'},
        'Simulated Annealing': {'color': '#f39c12', 'marker': 'D'}
    }

    for r in results:
        method = r['method']
        gap = r['gap']
        time = r['execution_time']

        if any(x in method for x in ['Nearest', 'Savings', 'Insertion', 'Sweep']):
            family = 'Greedy'
        elif 'Local Search' in method:
            family = 'Local Search'
        elif 'Tabu' in method:
            family = 'Tabu Search'
        elif 'Annealing' in method:
            family = 'Simulated Annealing'
        else:
            continue

        ax.scatter(time, gap, c=families[family]['color'],
                   marker=families[family]['marker'], s=150,
                   alpha=0.7, edgecolors='black', linewidth=1.5)

        # Annoter les meilleures solutions
        if gap < 2 or time < 0.001:
            ax.annotate(method, (time, gap), fontsize=7,
                        xytext=(5, 5), textcoords='offset points')

    # Légende
    for family, props in families.items():
        ax.scatter([], [], c=props['color'], marker=props['marker'],
                   s=100, label=family, edgecolors='black', linewidth=1.5)

    ax.set_xlabel('Temps d\'exécution (secondes)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gap (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Front de Pareto: Qualité vs Temps - {instance_name}',
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Zone "meilleure"
    ax.axhspan(0, 5, alpha=0.1, color='green', label='Gap < 5%')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{instance_name}_pareto.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Graphique Pareto sauvegardé: {instance_name}_pareto.png")


def generate_all_plots(results_file):
    """Génère tous les graphiques"""
    print(f"\n{'=' * 60}")
    print(f"Génération des graphiques pour: {results_file}")
    print(f"{'=' * 60}\n")

    results = load_results(results_file)
    instance_name = results[0]['instance']

    plot_cost_comparison(results, instance_name)
    plot_time_comparison(results, instance_name)
    plot_gap_comparison(results, instance_name)
    plot_cost_vs_time(results, instance_name)
    plot_improvements_comparison(results, instance_name)
    plot_pareto_front(results, instance_name)

    print(f"\n{'=' * 60}")
    print(f" Tous les graphiques générés dans le dossier 'plots/'")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    import sys

    # Exemples d'utilisation
    files = [
        "results/demo_results.json",
        "results/A-n32-k5_results.json"
    ]

    for file in files:
        if Path(file).exists():
            generate_all_plots(file)
        else:
            print(f"⚠ Fichier non trouvé: {file}")
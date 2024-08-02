from django.http import JsonResponse
from .map_simulation import MapSimulation

def run_simulation(request):
    N = int(request.GET.get('N', 10))
    M = int(request.GET.get('M', 10))
    K = int(request.GET.get('K', 3))
    T = int(request.GET.get('T', 2))
    R = int(request.GET.get('R', 5))
    TIME = int(request.GET.get('TIME', 10))
    num_targets = int(request.GET.get('num_targets', 3))

    # Run the simulation
    simulation = MapSimulation(N, M, K, T, R, TIME, num_targets)
    simulation_data = simulation.run_simulation()

    return JsonResponse({"simulation_data": simulation_data})

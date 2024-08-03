from django.http import JsonResponse
from .map_simulation import run_simulation

def run_simulation_view(request):
    try:
        height = 5
        width = 5
        TIME = 10
        num_cities = 3
        num_missiles = 2
        num_towers = 1

        simulation_data = run_simulation(height, width, TIME, num_cities, num_missiles, num_towers)
        return JsonResponse(simulation_data, safe=False)

    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)

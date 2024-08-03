import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .map_simulation  import run_simulation

@csrf_exempt
def run_simulation_view(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            grid_size = int(data.get('gridSize', 5))
            num_towers = int(data.get('numTowers', 2))
            num_missiles = int(data.get('numMissiles', 2))
            num_cities = int(data.get('numCities', 2))

            # Run the simulation with the provided parameters
            simulation_result = run_simulation(height=grid_size, width=grid_size, TIME=10, num_cities=num_cities, num_missiles=num_missiles, num_towers=num_towers, max_iterations=4)

            return JsonResponse(simulation_result, safe=False)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=405)

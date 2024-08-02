# simulation_app/urls.py

from django.urls import path
from .views import run_simulation_view

urlpatterns = [
    path('run-simulation/', run_simulation_view, name='run_simulation'),
]

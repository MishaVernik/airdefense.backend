# simulation_app/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('run-simulation/', views.run_simulation, name='run_simulation'),
]

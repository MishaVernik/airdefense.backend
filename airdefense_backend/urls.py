from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('simulation_app.urls')),  # Include the URLs for the simulation app
]

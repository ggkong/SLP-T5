from django.urls import path
from subCellLoc import views

urlpatterns = [
    path('', views.showHtml),
    path('predict/', views.modelRun),
]
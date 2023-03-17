from django.urls import path
import model.views

urlpatterns = [
    path('uploadmodel', model.views.uploadmodel),
    path('getmodellist', model.views.modellist),
    path('integrity', model.views.integrity),
    path('risk', model.views.riskanalyse)
]

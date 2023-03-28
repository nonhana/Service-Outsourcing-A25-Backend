from django.urls import path
import model.views

urlpatterns = [
    path('uploadmodel', model.views.uploadmodel),
    path('getmodellist', model.views.modellist),
    path('modelinfo', model.views.modelinfo),
    path('integrity', model.views.integrity),
    path('risk', model.views.riskanalyse),
    path('upload', model.views.upload_file),
    path('update', model.views.updatemodel)
]

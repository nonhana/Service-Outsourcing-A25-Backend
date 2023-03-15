from django.urls import path
import user.views

urlpatterns = [
    path("getuserlist", user.views.userlist),
    path("register", user.views.register)
]

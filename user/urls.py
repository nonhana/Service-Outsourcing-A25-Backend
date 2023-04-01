from django.urls import path
import user.views

urlpatterns = [
    path("getuserlist", user.views.userlist),
    path("register", user.views.register),
    path('login', user.views.login),
    path('getuserinfo', user.views.getuserinfo),
    path('uploadphoto', user.views.uploadphoto),
    path('deletefile', user.views.deletefile),
    path('updateuserinfo', user.views.updateuserinfo)
]

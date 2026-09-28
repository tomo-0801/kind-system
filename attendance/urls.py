from django.urls import path
from . import views
 
app_name = "attendance"
 
urlpatterns = [
    path("", views.top, name="top"),
    path("register/", views.register_student, name="register"),
    path("camera/", views.camera, name="camera"),
    path("recognize/", views.recognize_face, name="recognize"),
    path("success/<int:record_id>/", views.success, name="success"),
    path("dashboard/", views.dashboard, name="dashboard"),
    # ... 既存のURL設定 ...
    path("students/", views.student_list, name="student_list"),
    path("students/<int:student_id>/delete/", views.delete_student, name="delete_student"),
    path("students/", views.student_list, name="student_list"),
    path("students/<int:student_id>/delete/", views.delete_student, name="delete_student"),
]

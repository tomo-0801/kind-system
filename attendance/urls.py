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

    # 園児一覧
    path("students/", views.student_list, name="student_list"),

    # 園児を非表示
    path("students/<int:student_id>/delete/", views.delete_student, name="delete_student"),

    # 非表示園児一覧
    path("students/inactive/", views.inactive_student_list, name="inactive_student_list"),

    # 非表示園児を復元
    path("students/<int:student_id>/restore/", views.restore_student, name="restore_student"),
]
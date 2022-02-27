from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("report", views.report, name="report"),
    path("feedback", views.feedback, name="feedback"),
    path("contact", views.contact, name="contact")
]
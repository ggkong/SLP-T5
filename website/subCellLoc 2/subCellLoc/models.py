from django.db import models

# Create your models here.


class UserProfile(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    organization = models.CharField(max_length=100)
    careers = models.TextField()

    def __str__(self):
        return self.name

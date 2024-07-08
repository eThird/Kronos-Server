from django.contrib import admin
from kronos.models import *

# Register your models here.
admin.site.register(Product)
admin.site.register(Sale)
admin.site.register(Customer)
admin.site.register(Sentiment)
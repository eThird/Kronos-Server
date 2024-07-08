from django.db import models

class Product(models.Model):
    name = models.CharField(max_length=100)
    price = models.DecimalField(max_digits=7, decimal_places=2)
    cost = models.DecimalField(max_digits=7, decimal_places=2)
    recommended_price = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

class Customer(models.Model):
    GENDER = [
        ("M", "Male"),
        ("F", "Female")
    ]

    name = models.CharField(max_length=100)
    age = models.IntegerField()
    gender = models.CharField(max_length=1, choices=GENDER)
    location = models.CharField(max_length=100)


class Sentiment(models.Model):
    SENTIMENT = [
        ("POS", "Positive"),
        ("NEG", "Negative"),
        ("NEU", "Neutral")
    ]

    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    comment = models.CharField(max_length=300)
    sentiment = models.CharField(max_length=3, choices=SENTIMENT, blank=True)
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE)


class Sale(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE)
    quantity = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    sales_date = models.DateField(auto_now_add=True) 
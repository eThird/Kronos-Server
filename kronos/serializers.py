from rest_framework import serializers
from .models import *
# from django.contrib.auth.models import User

# class UserSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = User
#         fields = ['id', 'username', 'email']

# class RegisterSerializer(serializers.Serializer):
#     username = serializers.CharField()
#     email = serializers.EmailField()
#     password = serializers.CharField()

#     def validate(self, data):
#         if User.objects.filter(username=data['username']).exists():
#             raise serializers.ValidationError("Username Already Exists!") 
        
#         if User.objects.filter(email=data['email']).exists():
#             raise serializers.ValidationError("Email Already Used!") 

#         return data
    
#     def create(self, validated_data):
#         user = User.objects.create(username=validated_data["username"], email=validated_data["email"])
#         user.set_password(validated_data["password"])
#         user.save()
#         return validated_data


# class LoginSerializer(serializers.Serializer):
#     username = serializers.CharField()
#     password = serializers.CharField()

class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = "__all__"

class CustomerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Customer
        fields = "__all__"

class SentimentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Sentiment
        fields = "__all__"

class SaleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Sale
        fields = "__all__"

class ProductSaleSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    name = serializers.CharField()
    sale_count = serializers.IntegerField()

class SalesDataSerializer(serializers.Serializer):
    date = serializers.ListField(child=serializers.DateField())
    sales = serializers.ListField(child=serializers.IntegerField())

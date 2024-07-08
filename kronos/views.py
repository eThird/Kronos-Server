from rest_framework.decorators import api_view
from rest_framework.views import APIView
from rest_framework import viewsets, status
from django.contrib.auth import authenticate
from rest_framework.authtoken.models import Token
    
from django.contrib.auth.models import User
from rest_framework.response import Response
from .models import *
from .serializers import *
from .utils import sentiment

# Testing Route
@api_view(["GET"])
def home(request):
    message = "Hello World! I am Live!"
    return Response(message)

@api_view(["GET"])
def get_product(request, product_id):
    product = Product.objects.get(id=product_id)
    serializer = ProductSerializer(product)
    return Response(serializer.data)

@api_view(["GET"])
def comments_by_prod(request):
    class SentimentSerializer(serializers.ModelSerializer):
        customer_name = serializers.CharField(source='customer.name', read_only=True)
        
        class Meta:
            model = Sentiment
            fields = ['id', 'customer_name', 'sentiment', 'product', 'comment']  # Include 'customer_name'

    sentiment = request.GET.get("sentiment")
    product = request.GET.get("product")

    if sentiment == "ALL":
        comments = Sentiment.objects.filter(product=product)
    else:
        comments = Sentiment.objects.filter(sentiment=sentiment, product=product)
    serializer = SentimentSerializer(comments, many=True)

    return Response(serializer.data)

class ProductAPI(APIView):
    def get(self, request):
        objs = Product.objects.all()
        serializer = ProductSerializer(objs, many=True)
        return Response(serializer.data)
    
    def post(self, request):
        data = request.data
        serializer = ProductSerializer(data = data)
        
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        
        return Response(serializer.errors)

    def patch(self, request):
        data = request.data
        obj = Product.objects.get(id = data["id"])
        serializer = ProductSerializer(obj, data = data, partial = True)

        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        
        return Response(serializer.errors)
    
    def delete(self, request):
        data = request.data
        obj = Product.objects.get(id = data["id"])
        obj.delete()

        return Response({"message": "Data Deleted", "data": data})

class CustomerViewSet(viewsets.ModelViewSet):
    serializer_class = CustomerSerializer
    queryset = Customer.objects.all()

class SentimentAPI(APIView):
    def get(self, request):
        objs = Sentiment.objects.all()
        serializer = SentimentSerializer(objs, many=True)
        return Response(serializer.data)
    
    def post(self, request):    
        data = request.data
        # Evaluating Sentiment of the Comment
        data["sentiment"] = sentiment.roberta_predict(data["comment"])
        serializer = SentimentSerializer(data = data)
        
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        
        return Response(serializer.errors)

    def patch(self, request):
        data = request.data
        obj = Sentiment.objects.get(id = data["id"])
        serializer = SentimentSerializer(obj, data = data, partial = True)

        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        
        return Response(serializer.errors)
    
    def delete(self, request):
        data = request.data
        obj = Sentiment.objects.get(id = data["id"])
        obj.delete()

        return Response({"message": "Data Deleted", "data": data})

class SaleViewSet(viewsets.ModelViewSet):
    serializer_class = SaleSerializer
    queryset = Sale.objects.all()

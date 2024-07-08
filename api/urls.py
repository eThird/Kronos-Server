from kronos.views import *
from dashboard.views import *

from rest_framework.routers import DefaultRouter
from django.urls import path, include

router = DefaultRouter()

router.register(r"customer", CustomerViewSet, basename="customer")
router.register(r"sale", SaleViewSet, basename="sale")

urlpatterns = router.urls

urlpatterns = [
    # General App Routes
    path("", include(router.urls)),
    path("customers/", get_customers),
    path("products/", ProductAPI.as_view()),
    path("product/<int:product_id>/", get_product),
    path("prod_comments/", comments_by_prod), # localhost:8000/api/prod_comments?sentiment=NEG&product=1
    path("sentiment/", SentimentAPI.as_view()),
     path('sales/',SalesPredictionView.as_view(), name='sales_prediction'),
    # Home Dashboard
    path("satisfaction/", satisfaction_rate),
    path("customer_count/",customer_count),
    path("total_revenue/", total_revenue),
    path("total_profit/", total_profit),
    path("gender_compo/", gender_composition),
    path("age_groups/", age_groups),
    path("latest_orders/", latest_orders),
    path("latest_products/", latest_products),
    path("high_perf_prods/", high_perf_prods), 
    path("low_perf_prods/", low_perf_prods), 
]

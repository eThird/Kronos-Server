from rest_framework.decorators import api_view
from rest_framework.views import APIView
from rest_framework.response import Response
from kronos.models import *
from django.db.models import Count, Sum
from kronos.serializers import *
from rest_framework import status
import joblib
import pandas as pd
import numpy as np

# Dashboard APIs
@api_view(["GET"])
def satisfaction_rate(request):
    pos_count = Sentiment.objects.filter(sentiment="POS").count()
    neg_count = Sentiment.objects.filter(sentiment="NEG").count()
    rate = (pos_count*100)/(pos_count+neg_count)

    return Response({"satisfaction_rate": round(rate, 2)})

@api_view(["GET"])
def customer_count(request):
    count = Customer.objects.count()
    return Response({
        "customer_count": count
    })

@api_view(["GET"])
def total_revenue(request):
    revenue=0
    all_sales = Sale.objects.all()

    for sale in all_sales:
        product_price = sale.product.price
        revenue+=(product_price*sale.quantity)

    return Response({
        "revenue": revenue
    })


@api_view(["GET"])
def total_profit(request):
    profit=0
    all_sales = Sale.objects.all()

    for sale in all_sales:
        product_price = sale.product.price
        prduct_cost = sale.product.cost
        profit+=((product_price-prduct_cost)*sale.quantity)

    return Response({
        "profit": profit
    })

@api_view(["GET"])
def gender_composition(request):
    male_count = Customer.objects.filter(gender="M").count()
    female_count = Customer.objects.filter(gender="F").count()
    total_customers = Customer.objects.count()

    return Response({
        "male_compo": (male_count*100)/total_customers,
        "female_compo": (female_count*100)/total_customers,
        "others_compo": ((total_customers-male_count-female_count)*100)/total_customers
    })

@api_view(["GET"])
def age_groups(request):
    under_25 = Customer.objects.filter(age__lte=25).count()
    under_40 = Customer.objects.filter(age__lte=40, age__gte=26).count()
    under_75 = Customer.objects.filter(age__lte=75, age__gte=41).count()

    return Response({
        "under_25": round((under_25*100)/(under_25+under_40+under_75), 2),
        "under_40": round((under_40*100)/(under_25+under_40+under_75), 2),
        "under_75": round((under_75*100)/(under_25+under_40+under_75), 2)
    })

@api_view(["GET"])
def latest_orders(request):
    latest_orders = Sale.objects.order_by('-created_at')[:6]
    orders = []

    for order in latest_orders:
        orders.append({
                        "product": order.product.name,
                       "customer": order.customer.name,
                       "created_at": str(order.created_at)[:11],
                        "quantity": order.quantity
                       })

    return Response(orders)

@api_view(["GET"])
def latest_products(request):
    latest_products = Product.objects.order_by('-created_at')[:6]
    serializer = ProductSerializer(latest_products, many=True)

    return Response(serializer.data)


@api_view(["GET"])
def high_perf_prods(request): 
    # Perform left join and aggregation to get highest selling products
    highest_selling_products = Product.objects.annotate(
        sale_count=Sum('sale__quantity')
    ).order_by('-sale_count')

    # Serialize the data
    serializer = ProductSaleSerializer(highest_selling_products, many=True)

    return Response(serializer.data)

@api_view(["GET"])
def low_perf_prods(request): 
        # Perform left join and aggregation to get least selling products
        least_selling_products = Product.objects.annotate(
            sale_count=Sum('sale__quantity')
        ).order_by('sale_count')

        # Serialize the data
        serializer = ProductSaleSerializer(least_selling_products, many=True)

        return Response(serializer.data)

@api_view(["GET"])
def get_customers(request):
    customers = Customer.objects.all()
    serializer = CustomerSerializer(customers, many=True)

    return Response(serializer.data)

# Load the trained model and scaler
lr_model = joblib.load('models/linear_regression_model.pkl')
scaler = joblib.load('models/scaler.pkl')

class SalesPredictionView(APIView):
    def post(self, request):
        serializer = SalesDataSerializer(data=request.data)
        if serializer.is_valid():
            dates = serializer.validated_data['date']
            sales = serializer.validated_data['sales']

            # Convert input data to DataFrame
            data = pd.DataFrame({'date': pd.to_datetime(dates), 'sales': sales})
            data.set_index('date', inplace=True)

            # Feature engineering
            data['sales_diff'] = data['sales'].diff()
            data.dropna(inplace=True)

            # Create supervised data
            def create_supervised(data, lag=1):
                df = pd.DataFrame(data)
                columns = [df.shift(i) for i in range(1, lag+1)]
                columns.append(df)
                df = pd.concat(columns, axis=1)
                df.fillna(0, inplace=True)
                return df

            supervised_data = create_supervised(data['sales_diff'], 12)
            test_data = supervised_data[-12:]

            # Scaling features
            test_data = scaler.transform(test_data)
            x_test = test_data[:, 1:]
            y_test = test_data[:, 0]

            # Predict
            lr_predict = lr_model.predict(x_test)

            # Inverse transform to original scale
            lr_predict = scaler.inverse_transform(np.concatenate((lr_predict.reshape(-1, 1), x_test), axis=1))[:, 0]

            # Add the last actual sales value to the predictions to get the cumulative sales
            if len(data['sales']) >= 13:
                last_actual_sales = data['sales'].values[-13]
            else:
                last_actual_sales = data['sales'].values[0]

            lr_predict_cumulative = np.cumsum(np.insert(lr_predict, 0, last_actual_sales))[1:]

            # Actual sales for the last 12 months
            actual_sales = data['sales'].values[-12:]  # Adjusted to get the last 12 actual sales values

            response_data = {
                'actual_sales': actual_sales.tolist(),
                'predicted_sales': lr_predict_cumulative.tolist()
            }
            return Response(response_data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
 
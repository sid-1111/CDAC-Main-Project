from flask import Flask, render_template, request
import os
import sys
import datetime

# Add current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your modules
from sentiment_analyzer import SentimentAnalyzer
from clustering_models import ClusteringModels
from delivery_analyzer import DeliveryAnalyzer
# from sales_forecaster import SalesForecaster

app = Flask(__name__)

# Jinja filter to format currency
@app.template_filter('format_currency')
def format_currency_filter(value, currency_symbol='₹'):
    try:
        return f"{currency_symbol}{float(value):,.2f}"
    except (ValueError, TypeError):
        return f"{currency_symbol}N/A"

# Model instances (global)
sentiment_analyzer_obj = None
clustering_models_obj = None
delivery_analyzer_obj = None
sales_forecaster_obj = None

# Load all models on startup
def load_all_models_on_startup():
    global sentiment_analyzer_obj, clustering_models_obj, delivery_analyzer_obj, sales_forecaster_obj

    print("Loading all models...")

    try:
        sentiment_analyzer_obj = SentimentAnalyzer(
            s3_bucket_name='ecom-models-007',
            s3_model_key_prefix=''
        )
        print("SentimentAnalyzer loaded from S3.")
    except Exception as e:
        print(f"S3 load failed: {e}")
        try:
            sentiment_analyzer_obj = SentimentAnalyzer(model_name_or_path="./sentiment_model")
            print("SentimentAnalyzer loaded locally.")
        except Exception as fallback:
            print(f"Fallback failed: {fallback}")

    try:
        clustering_models_obj = ClusteringModels(
            seller_model_path="models/seller_clustering_model.pkl",
            review_model_path="models/review_clustering_model.pkl",
            customer_model_path="models/customer_clustering_model.pkl"
        )
        print("Clustering models loaded.")
    except Exception as e:
        print(f"Clustering load error: {e}")

    try:
        delivery_analyzer_obj = DeliveryAnalyzer(precomputed_risk_data_path=None)
        print("DeliveryAnalyzer loaded.")
    except Exception as e:
        print(f"DeliveryAnalyzer load error: {e}")

    # Sales forecaster loading (if implemented later)
    # try:
    #     sales_forecaster_obj = SalesForecaster(model_path="models/sales_forecasting_model.pkl")
    #     print("SalesForecaster loaded.")
    # except Exception as e:
    #     print(f"Sales forecaster error: {e}")

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Sentiment analysis
@app.route('/sentiment', methods=['GET', 'POST'])
def sentiment_route():
    result = None
    if request.method == 'POST' and sentiment_analyzer_obj:
        text_input = request.form.get('review_text')
        result = sentiment_analyzer_obj.analyze_sentiment(text_input)
    elif request.method == 'POST':
        result = {"error": "Sentiment analysis model not available."}
    return render_template('customer_emotion_analysis.html', result=result)

# ✅ Delivery delay prediction (Updated)
@app.route('/delivery', methods=['GET', 'POST'])
def delivery_route():
    result = None
    if request.method == 'POST':
        if delivery_analyzer_obj:
            try:
                region_input = request.form.get('region')
                seller_input = request.form.get('seller_id')
                estimated_days = int(request.form.get('delivery_estimated_days'))

                analysis_result = delivery_analyzer_obj.analyze_delivery_risk(
                    region_input, seller_input, estimated_days
                )

                result = {
                    "region": region_input,
                    "seller_id": seller_input,
                    "delivery_estimated_days": estimated_days,
                    "risk_level": analysis_result.get("risk_level", "Unknown"),
                    "prediction_message": analysis_result.get("prediction_message", "No prediction."),
                    "notes": analysis_result.get("notes", "")
                }
            except Exception as e:
                result = {"error": f"Error analyzing delivery risk: {e}"}
        else:
            result = {"error": "Delivery analysis model not loaded."}
    return render_template('delivery_delay_prevention.html', result=result)

# Clustering route
@app.route('/clustering', methods=['GET', 'POST'])
def clustering_route():
    segment_type = request.args.get('type', 'customer')
    result = None

    if request.method == 'POST' and clustering_models_obj:
        try:
            features_input = []
            if segment_type == 'seller':
                features_input = [float(request.form.get('seller_feature1', 0)), float(request.form.get('seller_feature2', 0))]
                cluster = clustering_models_obj.predict_seller_segment(features_input)
            elif segment_type == 'review':
                features_input = [float(request.form.get('review_sentiment_score', 0)), float(request.form.get('review_length', 0))]
                cluster = clustering_models_obj.predict_review_segment(features_input)
            elif segment_type == 'customer':
                features_input = [
                    float(request.form.get('customer_feature1', 0)),
                    float(request.form.get('customer_feature2', 0)),
                    float(request.form.get('customer_feature3', 0))
                ]
                cluster = clustering_models_obj.predict_customer_segment(features_input)
            else:
                return {"error": "Invalid segment type."}
            result = {
                "type": segment_type.capitalize(),
                "input_features": features_input,
                "cluster": cluster
            }
        except Exception as e:
            result = {"error": f"Clustering error: {e}"}
    elif request.method == 'POST':
        result = {"error": "Clustering models not available."}

    return render_template('buying_pattern_recognition.html', result=result, segment_type=segment_type)

# Forecasting (optional)
@app.route('/forecasting', methods=['GET', 'POST'])
def forecasting_route():
    result = None
    if request.method == 'POST':
        try:
            periods = int(request.form.get('periods'))
            forecast = []
            for i in range(periods):
                sales_value = 10000 + (i * 500)
                forecast.append((f"Period {i+1}", sales_value))
            result = {
                "periods": periods,
                "forecast": forecast
            }
        except Exception as e:
            result = {"error": f"Forecasting error: {e}"}
    return render_template('accurate_sales_forecasting.html', result=result)

# Static routes
@app.route('/team')
def team_route():
    return render_template('team.html')

@app.route('/support')
def support_route():
    return render_template('support.html')

# --- Launch app ---
if __name__ == '__main__':
    if not os.path.exists('models'):
        os.makedirs('models')

    load_all_models_on_startup()
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

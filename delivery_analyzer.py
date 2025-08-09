class DeliveryAnalyzer:
    def __init__(self, precomputed_risk_data_path=None):
        self.risk_data = None  # Optional: Load data if needed

    def analyze_delivery_risk(self, region, seller_id, estimated_days):
        # Simple logic – Replace with real logic or ML model later
        region = region.lower()
        if region in ['mumbai', 'delhi'] and estimated_days > 10:
            return {
                "risk_level": "High",
                "prediction_message": "High delay risk. Consider early shipment.",
                "notes": f"Seller '{seller_id}' in '{region}' often has delays beyond {estimated_days} days."
            }
        elif estimated_days > 7:
            return {
                "risk_level": "Medium",
                "prediction_message": "Moderate delivery risk.",
                "notes": "Delivery slightly above normal duration."
            }
        else:
            return {
                "risk_level": "Low",
                "prediction_message": "Delivery expected on time.",
                "notes": "No delay indicators found."
            }

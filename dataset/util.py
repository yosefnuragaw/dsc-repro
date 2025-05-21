import pandas as pd
from sklearn.model_selection import train_test_split
import os
from datetime import datetime

def install_hotel_booking_dataset():
    df = pd.read_csv( os.path.join("dataset", "hotel_booking.csv"))
    df.drop(columns=["name","email","phone-number","credit_card","reservation_status","reservation_status_date"],inplace=True)
    top_countries = ["PRT", "GBR", "FRA", "ESP", "DEU"] #Top 5 country

    for country in top_countries:    
        df_country = df[df["country"] == country].copy()  
        df_country = df_country.drop(columns=["country"])  
        train, test = train_test_split(df_country, test_size=0.2, random_state=42, stratify=df_country["is_canceled"])
        
        train.to_csv(os.path.join("dataset", f"{country}_D.csv") , index=False)
        test.to_csv(os.path.join("dataset", f"{country}_B.csv") , index=False)

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] COUNTRY {country} \t LOADED")
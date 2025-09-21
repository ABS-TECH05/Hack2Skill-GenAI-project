#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install google-cloud-vision pandas')


# In[ ]:


import pandas as pd
import random

random.seed(42)

# Couriers and services
couriers = ["DTDC","Delhivery","Blue Dart","Ecom Express","India Post","FedEx","DHL","Shiprocket"]
services = {
    "DTDC": ["Express","Surface"],
    "Delhivery": ["Surface","Express"],
    "Blue Dart": ["Express"],
    "Ecom Express": ["Surface","Economy"],
    "India Post": ["Speed Post","Economy"],
    "FedEx": ["Express","Economy","International Express"],
    "DHL": ["Express","International Express"],
    "Shiprocket": ["Aggregator"]
}

origin_zones = ["Local","Zone A","Zone B","Zone C","Zone D","National","International"]
dest_zones = origin_zones.copy()

# Weight slabs (0–500 g, ..., 9500–10000 g)
weights = [(i, i+500) for i in range(0,10000,500)]

# Surcharge pool
surcharge_pool = [
    "Fuel surcharge", "Fuel surcharge; COD charges", "Fuel surcharge; Remote area charge",
    "Dimensional weight applies", "Insurance optional", "No extra surcharges (retail)"
]

# Dimensional weight factor (for volumetric pricing)
def dimensional_weight_factor(service):
    if "International" in service or "Express" in service:
        return round(random.uniform(1.1, 1.5), 2)
    else:
        return 1.0

# Price model
def price_model(courier, service, origin, dest, w_start, w_end):
    courier_factor = {"DTDC":1.0,"Delhivery":1.1,"Blue Dart":1.4,"Ecom Express":0.9,
                      "India Post":0.6,"FedEx":2.2,"DHL":2.5,"Shiprocket":0.8}[courier]
    service_mult = 1.0
    if "Express" in service: service_mult = 1.4
    if "International" in service: service_mult = 3.5
    if "Economy" in service: service_mult = 0.85
    if "Aggregator" in service: service_mult = 0.9
    zone_order = {"Local":1, "Zone A":1.2, "Zone B":1.5, "Zone C":1.8, "Zone D":2.1,
                  "National":1.6,"International":4.0}
    dist_mult = (zone_order.get(origin,1)+zone_order.get(dest,1))/2.0
    w_mid = (w_start + w_end)/2.0
    base_small = 25.0
    weight_mult = 1 + (w_mid/1000.0)**1.05
    raw = base_small * courier_factor * service_mult * dist_mult * weight_mult
    noise = random.uniform(0.9,1.1)
    base_price = max(10, round(raw * noise))
    add_price = max(5, round(base_price * 0.18 * random.uniform(0.9,1.2)))
    return int(base_price), int(add_price)

rows = []
target_rows = 1500
attempts = 0

while len(rows) < target_rows and attempts < target_rows * 10:
    attempts += 1
    courier = random.choice(couriers)
    service = random.choice(services[courier])
    origin = random.choice(origin_zones)
    if origin == "Local":
        dest = random.choices(dest_zones, weights=[40,15,10,8,5,15,7], k=1)[0]
    elif origin == "International":
        dest = random.choices(dest_zones, weights=[3,3,3,3,2,3,83], k=1)[0]
    else:
        dest = random.choices(dest_zones, weights=[5,20,20,18,12,20,5], k=1)[0]
    w_start, w_end = random.choice(weights)
    base_price, add_price = price_model(courier, service, origin, dest, w_start, w_end)
    surcharge = random.choice(surcharge_pool)
    dim_factor = dimensional_weight_factor(service)
    # international adjustment
    if "International" in service or origin=="International" or dest=="International":
        base_price = int(base_price * 1.6)
        add_price = int(add_price * 1.5)
    effective_date = "2025-09-20"
    source_url = {
        "DTDC":"https://www.dtdc.in/","Delhivery":"https://www.delhivery.com/",
        "Blue Dart":"https://www.bluedart.com/","Ecom Express":"https://www.ecomexpress.in/",
        "India Post":"https://www.indiapost.gov.in/","FedEx":"https://www.fedex.com/",
        "DHL":"https://www.dhl.com/","Shiprocket":"https://www.shiprocket.in/"
    }[courier]
    row = [courier,service,origin,dest,w_start,w_end,base_price,add_price,surcharge,effective_date,source_url,dim_factor]
    if row not in rows:
        rows.append(row)

df_full = pd.DataFrame(rows, columns=[
    "courier_name","service_type","origin_zone","dest_zone","weight_start_g","weight_end_g",
    "base_price_inr","additional_price_inr","surcharge_rule","effective_date","source_url",
    "dimensional_weight_factor"
])

# Save CSV
df_full.to_csv("courier_rates_all_full.csv", index=False)
print("✅ Saved courier_rates_all_full.csv with", len(df_full), "rows")


# In[ ]:


from google.cloud import vision
import io

client = vision.ImageAnnotatorClient()

def image_to_text(image_path):
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description if texts else ""


# In[ ]:


get_ipython().system('pip uninstall -y google-cloud')
get_ipython().system('pip install --quiet --upgrade google-cloud-vision')


# In[ ]:


from google.cloud import vision
import io

client = vision.ImageAnnotatorClient()


# In[ ]:


from google.colab import files
uploaded = files.upload()

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/content/service_account.json"


# In[ ]:


get_ipython().system('pip install --quiet googlemaps')


# In[ ]:


from google.colab import files
uploaded = files.upload()


# In[ ]:


with open("maps_api_key.txt", "r") as f:
    MAPS_API_KEY = f.read().strip()

print("Key loaded, length:", len(MAPS_API_KEY))


# In[ ]:


import googlemaps

with open("maps_api_key.txt", "r") as f:
    MAPS_API_KEY = f.read().strip()

gmaps = googlemaps.Client(key=MAPS_API_KEY)


# In[ ]:


def distance_km_from_pincodes(src_pincode, dst_pincode):
    result = gmaps.distance_matrix(
        origins=[src_pincode],
        destinations=[dst_pincode],
        mode="driving"
    )

    elem = result["rows"][0]["elements"][0]
    if elem.get("status") != "OK":
        raise Exception(f"Error: {elem.get('status')}")
    meters = elem["distance"]["value"]
    return meters / 1000.0


# In[ ]:


import requests
from bs4 import BeautifulSoup

def get_dtdc_price(src_pincode, dst_pincode, weight):
    url = "https://www.dtdc.in/calculate-rate.aspx"
    payload = {
        "fromPincode": src_pincode,
        "toPincode": dst_pincode,
        "weight": weight,
    }
    res = requests.post(url, data=payload)
    soup = BeautifulSoup(res.text, "html.parser")
    price = soup.find("span", {"id": "total_price"}).text
    return price


# In[ ]:


import requests

def get_delhivery_price(src_pincode, dst_pincode, weight):
    url = "https://track.delhivery.com/api/kinko/v1/invoice/charges/.json"
    payload = {
        "from_pincode": src_pincode,
        "to_pincode": dst_pincode,
        "cod": 0,
        "weight": weight * 1000  # in grams
    }
    headers = {"Authorization": "Token <PUBLIC_DEMO_TOKEN>"}
    res = requests.get(url, params=payload, headers=headers)
    return res.json()


# In[ ]:


def get_bluedart_price(src_pincode, dst_pincode, weight):
    return "Bluedart API requires login, scraping not possible."


# In[ ]:


def get_india_post_price(src_pincode, dst_pincode, weight):
    url = "https://www.indiapost.gov.in/VAS/Pages/CalculatePostage.aspx"
    payload = {
        "from": src_pincode,
        "to": dst_pincode,
        "weight": weight
    }
    res = requests.post(url, data=payload)
    return res.text


# In[ ]:


def get_dtdc_price(src, dst, weight):
    return 50 + 25*weight + abs(int(src)-int(dst))/100

def get_delhivery_price(src, dst, weight):
    return 40 + 20*weight + abs(int(src)-int(dst))/90

def get_india_post_price(src, dst, weight):
    return 30 + 15*weight + abs(int(src)-int(dst))/120


# In[ ]:


def get_all_prices(src_pincode, dst_pincode, weight):
    results = {}

    try:
        results["DTDC"] = get_dtdc_price(src_pincode, dst_pincode, weight)
    except Exception as e:
        results["DTDC"] = f"Error: {e}"

    try:
        results["Delhivery"] = get_delhivery_price(src_pincode, dst_pincode, weight)
    except Exception as e:
        results["Delhivery"] = f"Error: {e}"

    try:
        results["India Post"] = get_india_post_price(src_pincode, dst_pincode, weight)
    except Exception as e:
        results["India Post"] = f"Error: {e}"

    results["Bluedart"] = "API login required"
    results["XpressBees"] = "API login required"

    return results


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

np.random.seed(42)
n_samples = 500
weights = np.random.uniform(0.1, 20, n_samples)
distances = np.random.uniform(10, 2000, n_samples)
companies = np.random.choice(['DTDC','Delhivery','FedEx','Ecom Express'], n_samples)

company_base = {'DTDC':50, 'Delhivery':40, 'FedEx':100, 'Ecom Express':60}
company_wrate = {'DTDC':25, 'Delhivery':20, 'FedEx':50, 'Ecom Express':30}
company_drate = {'DTDC':0.2, 'Delhivery':0.25, 'FedEx':0.5, 'Ecom Express':0.3}

prices = []
for w,d,c in zip(weights, distances, companies):
    price = company_base[c] + w*company_wrate[c] + d*company_drate[c] + np.random.normal(0,10)
    prices.append(price)

df = pd.DataFrame({
    'weight_kg': weights,
    'distance_km': distances,
    'company': companies,
    'price': prices
})

df_enc = pd.get_dummies(df, columns=['company'])

X = df_enc.drop('price', axis=1)
y = df_enc['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Model R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

feature_cols = list(X.columns)


# In[ ]:


import gradio as gr
import io

def courier_price_estimator(image, source_pin, dest_pin):

    np.random.seed(0)
    est_weight = round(np.random.uniform(0.5, 5.0), 2)
    try:
        dist = abs(int(source_pin) - int(dest_pin)) / 10
    except:
        dist = 100
    results = []
    for c in ['DTDC','Delhivery','FedEx','Ecom Express']:
        sample = pd.DataFrame([{
            'weight_kg': est_weight,
            'distance_km': dist,
            'company_'+c:1,
            **{f'company_{other}':0 for other in ['DTDC','Delhivery','FedEx','Ecom Express'] if other!=c}
        }])
        price = model.predict(sample[feature_cols])[0]
        results.append([c, round(price,2)])

    df_out = pd.DataFrame(results, columns=["Courier Company","Estimated Price (INR)"])
    cheapest = df_out.loc[df_out["Estimated Price (INR)"].idxmin()]
    return f"Estimated Weight: {est_weight} kg. Cheapest: {cheapest['Courier Company']} @ ₹{cheapest['Estimated Price (INR)']}", df_out


with gr.Blocks() as demo:
    gr.Markdown("# 📦 GenAI Courier Price Estimator (Prototype)")
    with gr.Row():
        img = gr.Image(type="pil", label="Upload picture of items")
        with gr.Column():
            src = gr.Textbox(label="Source Pincode")
            dst = gr.Textbox(label="Destination Pincode")
            btn = gr.Button("Estimate Courier Prices")
    out1 = gr.Textbox(label="Summary")
    out2 = gr.Dataframe(headers=["Courier Company","Estimated Price (INR)"])

    btn.click(courier_price_estimator, inputs=[img,src,dst], outputs=[out1,out2])

demo.launch(share=True)


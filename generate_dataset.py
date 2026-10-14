"""Generates a large synthetic-but-realistic crop dataset (data/crop_data.csv).
Run:  python generate_dataset.py
"""
import numpy as np, pandas as pd, os

rng = np.random.default_rng(42)
N = 30000

# crop: (rain_mean, rain_sd, temp_mean, temp_sd, base_yield t/ha, season, {soil: weight})
CROPS = {
 "Rice":      (1300, 220, 26, 2.2, 4.2, "Kharif", {"Alluvial":5,"Clay":4,"Loamy":2,"Black":1,"Red":1,"Sandy":0.2}),
 "Wheat":     (550,  110, 17, 2.5, 3.6, "Rabi",   {"Alluvial":5,"Loamy":4,"Black":2,"Clay":1.5,"Red":0.7,"Sandy":0.5}),
 "Maize":     (800,  140, 23, 2.5, 3.8, "Kharif", {"Loamy":5,"Alluvial":3,"Red":2.5,"Black":1.5,"Sandy":1,"Clay":0.7}),
 "Cotton":    (750,  120, 29, 2.0, 2.2, "Kharif", {"Black":6,"Alluvial":2,"Red":1,"Loamy":1,"Clay":0.5,"Sandy":0.2}),
 "Sugarcane": (1700, 260, 30, 2.0, 70,  "Annual", {"Alluvial":4,"Black":3.5,"Loamy":3,"Clay":1.5,"Red":1,"Sandy":0.2}),
 "Groundnut": (520,  90,  27, 2.0, 1.9, "Kharif", {"Sandy":5,"Red":4,"Loamy":2,"Alluvial":0.8,"Black":0.5,"Clay":0.2}),
 "Millet":    (380,  70,  31, 2.2, 1.4, "Kharif", {"Sandy":5,"Red":3.5,"Loamy":1,"Black":0.8,"Alluvial":0.5,"Clay":0.2}),
 "Soybean":   (1000, 150, 25, 2.0, 2.4, "Kharif", {"Black":5,"Loamy":3,"Alluvial":2,"Red":1.2,"Clay":1,"Sandy":0.3}),
 "Chickpea":  (420,  80,  21, 2.0, 1.6, "Rabi",   {"Black":4,"Loamy":3,"Sandy":2,"Red":2,"Alluvial":1.5,"Clay":0.5}),
 "Potato":    (650,  110, 15, 2.0, 22,  "Rabi",   {"Loamy":5,"Sandy":3,"Alluvial":3,"Red":1,"Black":0.5,"Clay":0.5}),
 "Jute":      (1650, 200, 28, 1.8, 2.5, "Kharif", {"Alluvial":6,"Clay":3,"Loamy":2,"Black":0.3,"Red":0.2,"Sandy":0.1}),
 "Tea":       (2100, 280, 21, 1.8, 2.1, "Annual", {"Red":5,"Loamy":3,"Sandy":1,"Alluvial":0.7,"Clay":0.5,"Black":0.2}),
}
SOILS = ["Alluvial","Black","Red","Sandy","Clay","Loamy"]
SOIL_PH = {"Alluvial":(7.0,.5),"Black":(7.8,.5),"Red":(5.8,.5),"Sandy":(6.5,.6),"Clay":(7.2,.6),"Loamy":(6.8,.4)}
STATES = ["Andhra Pradesh","Telangana","Punjab","Uttar Pradesh","Maharashtra","Gujarat","Karnataka",
          "Tamil Nadu","West Bengal","Bihar","Madhya Pradesh","Rajasthan","Odisha","Assam"]

names = list(CROPS)
rows = []
crop_pick = rng.choice(names, size=N)
for crop in crop_pick:
    rm, rs, tm, ts, by, season, sw = CROPS[crop]
    soil = rng.choice(list(sw), p=np.array(list(sw.values()))/sum(sw.values()))
    rain = max(150, rng.normal(rm, rs*0.8))
    temp = rng.normal(tm, ts*0.8)
    # suitability 0..1 (gaussian in rain / temp, soil weight)
    s_rain = np.exp(-0.5*((rain-rm)/(rs*1.4))**2)
    s_temp = np.exp(-0.5*((temp-tm)/(ts*1.4))**2)
    s_soil = 0.55 + 0.45*sw[soil]/max(sw.values())
    suit = s_rain*s_temp*s_soil
    yld = by*(0.35 + 0.85*suit) * rng.normal(1, 0.06)
    ph = rng.normal(*SOIL_PH[soil])
    rows.append((rng.choice(STATES), int(rng.integers(2005, 2025)), season, crop, soil,
                 round(rain,1), round(temp,1), round(ph,2), round(float(rng.uniform(35,90)),1),
                 round(max(yld,0.05),2)))

df = pd.DataFrame(rows, columns=["state","year","season","crop","soil_type","rainfall_mm",
                                 "avg_temp_c","soil_ph","humidity_pct","yield_t_ha"])
os.makedirs("data", exist_ok=True)
df.to_csv("data/crop_data.csv", index=False)
print(df.shape); print(df.head())

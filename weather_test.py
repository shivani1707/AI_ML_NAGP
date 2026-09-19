# quick_weather_test.py
import requests

url = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=1.3521&longitude=103.8198"
    "&current=temperature_2m,precipitation,weather_code"
    "&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code"
    "&forecast_days=3"
    "&timezone=Asia%2FSingapore"
)
response = requests.get(url)
print(response.json())
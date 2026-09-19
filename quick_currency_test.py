# quick_currency_test.py
import requests

url = "https://api.frankfurter.dev/v1/latest?base=INR&symbols=SGD"
response = requests.get(url)
print(response.json())
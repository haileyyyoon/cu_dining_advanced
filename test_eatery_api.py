import requests
import json

url = "https://admin-now.dining.cornell.edu/api/1.0/dining/eateries.json"

headers = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

response = requests.get(url, headers=headers)
print("Status:", response.status_code)
print("Body (first 1000 chars):", response.text[:1000])

data = response.json()
eateries = data["data"]["eateries"]
print(f"\nFound {len(eateries)} eateries.\n")
print(json.dumps(eateries[0], indent=2)[:2000])

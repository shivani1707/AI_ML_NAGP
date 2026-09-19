from mcp.server.fastmcp import FastMCP
import requests

mcp = FastMCP("Weather")

# Singapore's coordinates (since this project is scoped to one destination)
SINGAPORE_LAT = 1.3521
SINGAPORE_LON = 103.8198

@mcp.tool()
def get_weather_forecast(days: int = 3) -> str:
    """
    Get the current weather and forecast for Singapore.
    Args:
        days: Number of forecast days to return (1-7). Default is 3.
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={SINGAPORE_LAT}&longitude={SINGAPORE_LON}"
        "&current=temperature_2m,precipitation,weather_code"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code"
        f"&forecast_days={days}"
        "&timezone=Asia%2FSingapore"
    )
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    current = data["current"]
    result = f"Current temperature in Singapore: {current['temperature_2m']}C\n\n"

    result += "Forecast:\n"
    daily = data["daily"]
    for i, date in enumerate(daily["time"]):
        low = daily["temperature_2m_min"][i]
        high = daily["temperature_2m_max"][i]
        rain_chance = daily["precipitation_probability_max"][i]
        result += f"- {date}: {low} to {high} C, {rain_chance}% chance of rain\n"

    return result

if __name__ == "__main__":
    mcp.run(transport="stdio")
from mcp.server.fastmcp import FastMCP
import requests

mcp = FastMCP("Currency")

@mcp.tool()
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Convert an amount of money from one currency to another using current exchange rates.
    Args:
        amount: The amount of money to convert.
        from_currency: The 3-letter currency code to convert from (e.g. INR, USD, SGD).
        to_currency: The 3-letter currency code to convert to (e.g. INR, USD, SGD).
    """
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()

    url = f"https://api.frankfurter.dev/v1/latest?base={from_currency}&symbols={to_currency}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    rate = data["rates"][to_currency]
    converted = amount * rate

    return (
        f"{amount} {from_currency} = {converted:.2f} {to_currency} "
        f"(exchange rate: 1 {from_currency} = {rate} {to_currency}, as of {data['date']})"
    )

if __name__ == "__main__":
    mcp.run(transport="stdio")
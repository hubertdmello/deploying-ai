import requests
from openai import OpenAI
from __init__ import initialize_openai_client
from prompts import WEATHER_SYSTEM_PROMPT


def summarize_with_openai(text: str, client: OpenAI) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=300,
            messages=[
                {"role": "system", "content": WEATHER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Here's the weather data, please summarize it in a friendly way:\n\n{text}"
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error summarizing with OpenAI: {str(e)}"


def get_weather_info(location: str) -> str:
    if not location.strip():
        return "Please enter a city name."

    try:
        client = initialize_openai_client()

        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {"name": location, "count": 1, "language": "en", "format": "json"}
        geo_response = requests.get(geo_url, params=geo_params, timeout=5)
        geo_data = geo_response.json()

        if not geo_data.get("results"):
            return f"I couldn't find a city called '{location}'. Please try another name."

        result = geo_data["results"][0]
        latitude = result["latitude"]
        longitude = result["longitude"]
        city_name = result["name"]
        country = result.get("country", "")

        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh"
        }
        weather_response = requests.get(weather_url, params=weather_params, timeout=5)
        weather_data = weather_response.json()

        current = weather_data.get("current", {})
        temp = current.get("temperature_2m", "N/A")
        humidity = current.get("relative_humidity_2m", "N/A")
        wind_speed = current.get("wind_speed_10m", "N/A")
        weather_code = current.get("weather_code", 0)

        weather_descriptions = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
            55: "Dense drizzle", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow", 77: "Snow grains",
            80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
            85: "Slight snow showers", 86: "Heavy snow showers", 95: "Thunderstorm",
            96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
        }
        weather_desc = weather_descriptions.get(weather_code, "Unknown conditions")

        raw_weather_data = f"""Weather for {city_name}, {country}:
Temperature: {temp}°C
Humidity: {humidity}%
Wind Speed: {wind_speed} km/h
Conditions: {weather_desc}"""

        summarized_response = summarize_with_openai(raw_weather_data, client)

        return f"""## 🌍 Weather for {city_name}, {country}

{summarized_response}

---
*Data sourced from Open-Meteo API and summarized with OpenAI*"""

    except ValueError as e:
        return f"Configuration error: {str(e)}\n\nPlease ensure your OPENAI_API_KEY environment variable is set."
    except requests.exceptions.Timeout:
        return "The request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {str(e)}"
    except (KeyError, ValueError) as e:
        return f"Error processing weather data: {str(e)}"

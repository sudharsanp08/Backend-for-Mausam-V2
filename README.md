"""
Mausam v2 - Universal Backend API Gateway & Personalization Engine
Runs on: http://127.0.0.1:8000
"""
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
from pydantic import BaseModel
from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime, ForeignKey, Text, select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
import uvicorn

# ---------------------------------------------------------------------------
# 1. DATABASE CONFIGURATION
# ---------------------------------------------------------------------------
DATABASE_URL = "sqlite+aiosqlite:///./mausam_universal.db"
engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

class Location(Base):
    __tablename__ = "locations"
    id = Column(Integer, primary_key=True, index=True)
    city_name = Column(String(100), index=True, nullable=False)
    state = Column(String(100), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)

    weather_records = relationship("WeatherData", back_populates="location", cascade="all, delete-orphan")
    alerts = relationship("SevereAlert", back_populates="location", cascade="all, delete-orphan")

class WeatherData(Base):
    __tablename__ = "weather_data"
    id = Column(Integer, primary_key=True)
    location_id = Column(Integer, ForeignKey("locations.id"), nullable=False)
    recorded_at = Column(DateTime, default=datetime.utcnow)
    temperature_c = Column(Float, nullable=False)
    humidity_pct = Column(Float, nullable=False)
    aqi = Column(Integer, nullable=False)
    pollen_index = Column(Integer, nullable=False)
    uv_index = Column(Float, nullable=False)
    rainfall_prob_pct = Column(Float, nullable=False)
    soil_moisture_pct = Column(Float, nullable=False)

    location = relationship("Location", back_populates="weather_records")

class SevereAlert(Base):
    __tablename__ = "severe_alerts"
    id = Column(Integer, primary_key=True)
    location_id = Column(Integer, ForeignKey("locations.id"), nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    severity_level = Column(String(20), nullable=False)
    is_active = Column(Boolean, default=True)

    location = relationship("Location", back_populates="alerts")

# ---------------------------------------------------------------------------
# 2. SCHEMAS
# ---------------------------------------------------------------------------
class SevereAlertResponse(BaseModel):
    title: str
    description: str
    severity_level: str

class DynamicWidget(BaseModel):
    widget_id: str
    title: str
    priority: int
    data: dict
    explanation: str

class DashboardResponse(BaseModel):
    city_name: str
    state: str
    active_alerts: List[SevereAlertResponse]
    widgets: List[DynamicWidget]
    selected_persona: str

# ---------------------------------------------------------------------------
# 3. TELEMETRY FETCHING & GEOCODING (ALL INDIA)
# ---------------------------------------------------------------------------
async def fetch_live_indian_weather(lat: float, lon: float) -> dict:
    """Fetches real-time weather, UV, AQI, and calculates bio-pollen indices."""
    async with httpx.AsyncClient(timeout=6.0) as client:
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m,precipitation_probability,uv_index"
            "&hourly=soil_moisture_0_to_1cm&timezone=Asia%2FKolkata"
        )
        air_url = (
            f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}"
            "&current=pm2_5,pm10,european_aqi"
        )

        w_resp = await client.get(weather_url)
        a_resp = await client.get(air_url)

        w_data = w_resp.json().get("current", {})
        hourly = w_resp.json().get("hourly", {})
        a_data = a_resp.json().get("current", {})

        temp = w_data.get("temperature_2m", 30.0)
        humidity = w_data.get("relative_humidity_2m", 50.0)
        rain_prob = w_data.get("precipitation_probability", 10.0)
        uv = w_data.get("uv_index", 5.0)

        soil_list = hourly.get("soil_moisture_0_to_1cm", [0.35])
        soil_pct = round((soil_list[0] if soil_list else 0.35) * 100, 1)

        pm25 = a_data.get("pm2_5", 45.0)
        calculated_aqi = int(pm25 * 2.1) if pm25 else 75
        pollen = int(max(10, min(95, (temp * 2.2) - (humidity * 0.4) - (rain_prob * 0.3))))

        return {
            "temperature_c": temp,
            "humidity_pct": humidity,
            "aqi": calculated_aqi,
            "pollen_index": pollen,
            "uv_index": uv,
            "rainfall_prob_pct": float(rain_prob),
            "soil_moisture_pct": soil_pct
        }

async def geocode_indian_place(place_name: str) -> Optional[dict]:
    """Resolves any Indian location name into coordinates using Open-Meteo Geocoding."""
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={place_name}&count=5&language=en&format=json"
    async with httpx.AsyncClient(timeout=5.0) as client:
        res = await client.get(url)
        if res.status_code != 200:
            return None
        results = res.json().get("results", [])
        for place in results:
            if place.get("country_code", "").upper() == "IN":
                return {
                    "city_name": place.get("name"),
                    "state": place.get("admin1", "India"),
                    "latitude": place.get("latitude"),
                    "longitude": place.get("longitude")
                }
        return None

# ---------------------------------------------------------------------------
# 4. ADAPTIVE LAYOUT ENGINE (PERSONA RE-RANKING)
# ---------------------------------------------------------------------------
def build_adaptive_layout(weather_dict: dict, alerts: List[SevereAlert], persona: str):
    aqi_val = weather_dict["aqi"]
    pollen_val = weather_dict["pollen_index"]

    health_widget = DynamicWidget(
        widget_id="health_card",
        title="Air Quality & Respiratory Indices",
        priority=0,
        data={
            "AQI": f"{aqi_val} (Hazardous)" if aqi_val > 300 else (f"{aqi_val} (Poor)" if aqi_val > 150 else f"{aqi_val} (Moderate)"),
            "Pollen Index": f"{pollen_val}/100",
            "UV Index": f"{weather_dict['uv_index']}",
            "Humidity": f"{weather_dict['humidity_pct']}%"
        },
        explanation="Prioritized because your profile is Health & Environment."
    )

    agromet_widget = DynamicWidget(
        widget_id="agromet_card",
        title="Agromet Advisory & Soil Moisture",
        priority=0,
        data={
            "Soil Moisture": f"{weather_dict['soil_moisture_pct']}%",
            "Rainfall Probability": f"{weather_dict['rainfall_prob_pct']}%",
            "Sowing Window": "Favorable" if weather_dict['soil_moisture_pct'] > 45 else "Irrigation Advised"
        },
        explanation="Prioritized because your profile is Farmer."
    )

    commuter_widget = DynamicWidget(
        widget_id="commuter_card",
        title="Commuter Radar & Rain Forecast",
        priority=0,
        data={
            "Temperature": f"{weather_dict['temperature_c']}°C",
            "Precipitation Risk": f"{weather_dict['rainfall_prob_pct']}%",
            "Route Status": "Rain Expected - Delays Likely" if weather_dict['rainfall_prob_pct'] > 70 else "Clear / Low Delays"
        },
        explanation="Prioritized because your profile is Commuter / Traveler."
    )

    if persona == "farmer":
        agromet_widget.priority = 1
        commuter_widget.priority = 2
        health_widget.priority = 3
    elif persona == "commuter":
        commuter_widget.priority = 1
        health_widget.priority = 2
        agromet_widget.priority = 3
    else:  # health / default
        health_widget.priority = 1
        commuter_widget.priority = 2
        agromet_widget.priority = 3

    widgets = [health_widget, agromet_widget, commuter_widget]
    widgets.sort(key=lambda w: w.priority)

    alert_responses = [
        SevereAlertResponse(
            title=a.title,
            description=a.description,
            severity_level=a.severity_level
        ) for a in alerts if a.is_active
    ]

    return {
        "active_alerts": alert_responses,
        "widgets": widgets,
        "selected_persona": persona
    }

# ---------------------------------------------------------------------------
# 5. FASTAPI APPLICATION SETUP & ROUTING
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield

app = FastAPI(title="Mausam v2 Universal API", lifespan=lifespan)

# Allow cross-origin requests from the HTML frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/dashboard", response_model=DashboardResponse)
async def get_dashboard(city: str = "Chennai", persona: str = "health"):
    async with AsyncSessionLocal() as session:
        stmt = select(Location).where(Location.city_name.ilike(f"%{city}%"))
        res = await session.execute(stmt)
        location = res.scalars().first()

        if not location:
            geocoded = await geocode_indian_place(city)
            if not geocoded:
                raise HTTPException(status_code=404, detail=f"Location '{city}' could not be resolved within India.")
            
            location = Location(
                city_name=geocoded["city_name"],
                state=geocoded["state"],
                latitude=geocoded["latitude"],
                longitude=geocoded["longitude"]
            )
            session.add(location)
            await session.commit()
            await session.refresh(location)

        try:
            live_weather = await fetch_live_indian_weather(location.latitude, location.longitude)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Error fetching live weather telemetry: {str(e)}")

        alert_stmt = select(SevereAlert).where(
            SevereAlert.location_id == location.id,
            SevereAlert.is_active == True
        )
        alert_res = await session.execute(alert_stmt)
        alerts = alert_res.scalars().all()

        layout = build_adaptive_layout(live_weather, alerts, persona)

        return DashboardResponse(
            city_name=location.city_name,
            state=location.state,
            active_alerts=layout["active_alerts"],
            widgets=layout["widgets"],
            selected_persona=layout["selected_persona"]
        )

if __name__ == "__main__":
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)

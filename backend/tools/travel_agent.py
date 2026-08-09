"""
Travel Agent - הזמנת חופשות חכמה
משווה טיסות ומלונות, בוחר משתלם ומבצע הזמנה באישור
"""
import asyncio
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class Flight:
    airline: str
    from_city: str
    to_city: str
    departure: datetime
    arrival: datetime
    price: float
    currency: str
    stops: int
    duration_hours: float

@dataclass
class Hotel:
    name: str
    city: str
    price_per_night: float
    total_price: float
    rating: float
    reviews: int
    amenities: List[str]

class TravelAgent:
    """
    סוכן נסיעות - משווה ומזמין
    """
    def __init__(self):
        self.airlines = ["El Al", "Wizz Air", "Ryanair", "Turkish Airlines", "Lufthansa", "Arkia"]
        self.hotel_chains = ["Dan Hotels", "Isrotel", "Fattal", "Herods", "Hilton", "Booking.com Selection"]

    async def search_flights(self, from_city: str, to_city: str, date: str, return_date: str = None, passengers=1) -> List[Flight]:
        """חיפוש טיסות - מדמה API אמיתי של Skyscanner/Google Flights"""
        print(f"[Travel] ✈️ מחפש טיסות {from_city} -> {to_city} בתאריך {date}")
        
        # דמו ריאליסטי
        flights = []
        base_price = random.uniform(80, 400)
        
        for i in range(5):
            departure = datetime.strptime(date, "%Y-%m-%d") + timedelta(hours=random.randint(5, 22))
            duration = random.uniform(2.5, 6.0)
            arrival = departure + timedelta(hours=duration)
            stops = random.choices([0, 0, 0, 1, 1, 2], k=1)[0]
            
            price_mult = 1.0
            if stops == 0:
                price_mult = 1.3
            elif stops == 1:
                price_mult = 1.0
            else:
                price_mult = 0.8
            
            flights.append(Flight(
                airline=random.choice(self.airlines),
                from_city=from_city,
                to_city=to_city,
                departure=departure,
                arrival=arrival,
                price=round(base_price * price_mult * passengers * random.uniform(0.9, 1.1), 2),
                currency="$",
                stops=stops,
                duration_hours=round(duration, 1)
            ))
        
        flights.sort(key=lambda f: f.price)
        print(f"[Travel] נמצאו {len(flights)} טיסות, זולה: ${flights[0].price}")
        return flights

    async def search_hotels(self, city: str, check_in: str, check_out: str, guests=2) -> List[Hotel]:
        """חיפוש מלונות"""
        print(f"[Travel] 🏨 מחפש מלונות ב-{city} {check_in} -> {check_out}")
        
        check_in_dt = datetime.strptime(check_in, "%Y-%m-%d")
        check_out_dt = datetime.strptime(check_out, "%Y-%m-%d")
        nights = (check_out_dt - check_in_dt).days
        
        hotels = []
        for i in range(6):
            per_night = random.uniform(60, 250)
            rating = round(random.uniform(7.0, 9.6), 1)
            
            hotels.append(Hotel(
                name=f"{random.choice(self.hotel_chains)} {city} {i+1}",
                city=city,
                price_per_night=round(per_night, 2),
                total_price=round(per_night * nights, 2),
                rating=rating,
                reviews=random.randint(50, 2000),
                amenities=random.sample(["בריכה", "חדר כושר", "ארוחת בוקר", "WiFi חינם", "ספא", "חניה"], k=random.randint(2, 5))
            ))
        
        hotels.sort(key=lambda h: (h.total_price, -h.rating))
        return hotels

    async def find_best_deal(self, from_city: str, to_city: str, check_in: str, check_out: str, budget: float = None) -> Dict:
        """מציאת דיל משתלם - טיסה + מלון"""
        print(f"[Travel] 🎯 מחפש דיל משתלם {from_city}->{to_city}")
        
        flights_task = self.search_flights(from_city, to_city, check_in)
        hotels_task = self.search_hotels(to_city, check_in, check_out)
        
        flights, hotels = await asyncio.gather(flights_task, hotels_task)
        
        if not flights or not hotels:
            return {"success": False, "error": "לא נמצאו טיסות או מלונות"}
        
        # מצא קומבינציות
        deals = []
        for flight in flights[:3]:
            for hotel in hotels[:3]:
                total = flight.price + hotel.total_price
                score = (hotel.rating * 10) - (total / 100)  # ציון: דירוג גבוה + מחיר נמוך
                if budget and total > budget:
                    continue
                deals.append({
                    "flight": flight,
                    "hotel": hotel,
                    "total_price": round(total, 2),
                    "score": round(score, 2)
                })
        
        deals.sort(key=lambda d: (-d["score"], d["total_price"]))
        
        if not deals:
            return {"success": False, "error": f"לא נמצא דיל בתקציב ${budget}"}
        
        best = deals[0]
        
        return {
            "success": True,
            "from": from_city,
            "to": to_city,
            "check_in": check_in,
            "check_out": check_out,
            "best_deal": {
                "flight": {
                    "airline": best["flight"].airline,
                    "departure": best["flight"].departure.isoformat(),
                    "arrival": best["flight"].arrival.isoformat(),
                    "duration": best["flight"].duration_hours,
                    "stops": best["flight"].stops,
                    "price": best["flight"].price
                },
                "hotel": {
                    "name": best["hotel"].name,
                    "rating": best["hotel"].rating,
                    "reviews": best["hotel"].reviews,
                    "per_night": best["hotel"].price_per_night,
                    "total": best["hotel"].total_price,
                    "amenities": best["hotel"].amenities
                },
                "total_price": best["total_price"],
                "currency": "$",
                "score": best["score"]
            },
            "all_deals": [
                {
                    "flight_airline": d["flight"].airline,
                    "flight_price": d["flight"].price,
                    "hotel_name": d["hotel"].name,
                    "hotel_rating": d["hotel"].rating,
                    "total": d["total_price"]
                } for d in deals[:5]
            ],
            "requires_confirmation": True,
            "message": f"מצאתי דיל! ✈️ {best['flight'].airline} ב-${best['flight'].price} + 🏨 {best['hotel'].name} ({best['hotel'].rating}★) סה\"כ ${best['total_price']}. לאשר הזמנה בוס?"
        }

    async def book(self, deal: Dict, passenger_info: Dict) -> Dict:
        """הזמנה בפועל - דורש אישור"""
        print(f"[Travel] 📝 מבצע הזמנה עבור {passenger_info.get('name')}")
        
        return {
            "success": True,
            "booking_id": f"ADJ-{random.randint(100000, 999999)}",
            "passenger": passenger_info,
            "deal": deal,
            "status": "pending_confirmation",
            "message": f"הכנתי הזמנה {deal.get('from')}->{deal.get('to')} על שם {passenger_info.get('name')}. דורש אישור תשלום סופי שלך בוס! (לא מחייב בלי אישור)",
            "warning": "⚠️ זו הדגמה - לא בוצע חיוב אמיתי. באישורך אפתח דפדפן להזמנה אמיתית"
        }

def get_travel_agent():
    return TravelAgent()

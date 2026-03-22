import numpy as np
from utils.logger import log

def calculate_inventory(
    demand_mean,          
    demand_std,           
    lead_time=7,          
    unit_cost=100,        
    ordering_cost=500,    
    holding_rate=0.20     
):
    
    log("=== OPTIMIZATION STARTED ===")

    # 1️⃣ Annual Demand
    annual_demand = demand_mean * 365

    # 2️⃣ Holding Cost per Unit per Year
    holding_cost = unit_cost * holding_rate

    # 3️⃣ Economic Order Quantity (EOQ)
    # Prevent division by zero if holding cost is 0
    if holding_cost <= 0:
        eoq = 0
    else:
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)

    # 4️⃣ Safety Stock (95% Service Level)
    z = 1.65  # 95% service level
    safety_stock = z * demand_std * np.sqrt(lead_time)

    # 5️⃣ Reorder Point
    reorder_point = (demand_mean * lead_time) + safety_stock

    # 6️⃣ Total Annual Inventory Cost
    if eoq > 0:
        total_cost = ((annual_demand / eoq) * ordering_cost) + ((eoq / 2) * holding_cost)
    else:
        total_cost = 0

    return {
        "EOQ": round(eoq),
        "Safety Stock": round(safety_stock),
        "Reorder Point": round(reorder_point),
        "Annual Demand": round(annual_demand),
        "Total Annual Cost": round(total_cost)
    }

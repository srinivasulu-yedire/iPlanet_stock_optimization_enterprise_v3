import numpy as np
from utils.logger import log

def calculate_inventory(demand_mean, demand_std, lead_time=7, unit_cost=100, holding_cost=5):
    log("=== OPTIMIZATION STARTED ===")
    
    # Economic Order Quantity
    # 
    eoq = np.sqrt((2 * demand_mean * unit_cost) / holding_cost)
    
    # Safety Stock (95% service level)
    # 
    safety_stock = 1.65 * demand_std * np.sqrt(lead_time)
    
    # Reorder Point
    reorder_point = (demand_mean * lead_time) + safety_stock
    
    return eoq, safety_stock, reorder_point
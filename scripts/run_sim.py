import os
import sys
import traci
import csv

sumo_config = "mirage/config.sumocfg"
sumo_binary = "sumo-gui"

traci.start([sumo_binary, "-c", sumo_config])

with open("outputs/vehicle_log.csv", "w", newline="") as file:
    writer = csv.writer(file)
    # Added vx, vy, and angle for better precision
    writer.writerow(["time", "vehicle_id", "x", "y", "vx", "vy", "angle", "speed"])

    step = 0
    while step < 500:
        traci.simulationStep()
        vehicles = traci.vehicle.getIDList()

        for vehicle in vehicles:
            x, y = traci.vehicle.getPosition(vehicle)
            speed = traci.vehicle.getSpeed(vehicle)
            angle_deg = traci.vehicle.getAngle(vehicle)
            
            # Convert SUMO angle to velocity vectors
            import math
            angle_rad = math.radians(angle_deg)
            vx = speed * math.sin(angle_rad)
            vy = speed * math.cos(angle_rad)

            writer.writerow([step, vehicle, x, y, vx, vy, angle_deg, speed])

        step += 1

traci.close()
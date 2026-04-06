# D01 — AeroFleet X200 Battery Cooling System Overview
Document type: System overview  
Status: Reference  
Effective date: 2026-01-10

The AeroFleet X200 electric inspection drone uses a battery cooling subsystem called the Battery Cooling Unit (BCU).  
The BCU combines two axial cooling fans, a micro-pump, a narrow coolant loop, and a passive phase-change pad below the cell stack.

Key sensor tags used in maintenance logs:
- **BT-Top**: battery top-surface temperature
- **BT-Core**: estimated battery core temperature
- **Fan-I-L / Fan-I-R**: left and right fan current
- **Pump-F**: coolant flow estimate
- **Pack-ΔT**: temperature spread across the pack

Nominal operating guidance:
- Preferred **battery core temperature** during flight: **18–42 °C**
- Caution zone: **above 48 °C**
- Alarm zone: **above 52 °C**
- During fast charging, operators should try to keep battery core temperature **below 45 °C**

The thermal management objective is not only cooling. It is also to keep the pack temperature distribution uniform. A large thermal gradient across the battery pack can indicate blocked airflow, poor pad contact, or local cell degradation.

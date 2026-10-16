import numpy as np
import pandas as pd

# Parameters
num_ticks = 100000
x = np.linspace(0, 10, num_ticks)  # simulate x-axis from 0 to 10
np.random.seed(42)  # for reproducibility

# Base slopes (guessed from your graph)
slopes = {
    'Scatter': 24,
    'Look': 25,
    'Our_Algo': 27,
    'NstepLook': 29,
    'FCFS': 32
}

# Create power consumption data
power_data = {}

for algo, slope in slopes.items():
    base_line = slope * x  # linear increase
    noise = np.random.normal(0, 1, num_ticks)  # small random noise
    power = base_line + noise  # add noise
    power = np.maximum.accumulate(power)  # ensure non-decreasing (since power can't decrease)
    power_data[algo] = power

# Create DataFrame
df = pd.DataFrame({'tick': np.arange(1, num_ticks + 1)})
for algo in power_data:
    df[algo] = power_data[algo]

# Save to Excel if needed
df.to_excel('power_consumption_simulated.xlsx', index=False)

print("Power consumption dataset created and saved as 'power_consumption_simulated.xlsx'.")

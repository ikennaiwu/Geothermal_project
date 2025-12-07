# generate_realistic_geothermal.py
import pandas as pd
import numpy as np

np.random.seed(42)

def generate_realistic_iddp_habanero_data(n_samples=1000):
    """
    Generate 1000-row geothermal drilling dataset using ONLY:
    - IDDP depth-lithology logs (Elders et al. 2014, 2019)
    - Habanero operational ranges (Humphreys et al. 2014)
    - Bit wear and torque spike models from public dysfunction reports
    """
    depth = np.linspace(0, 5000, n_samples)
    lithology = []
    bit_condition = 1.0
    last_lith = None

    # Assign lithology per IDDP depth zones
    for d in depth:
        if d < 1200:
            lith = 0  # Basalt (Krafla shallow)
        elif 1200 <= d < 2400:
            lith = 2  # Rhyolite (magma intrusion)
        elif 2400 <= d < 3800:
            lith = 1  # Granite (Habanero-like deep)
        else:
            lith = 3  # Andesite (IDDP-2 bottom)
        lithology.append(lith)

    lithology = np.array(lithology)

    # WOB/RPM: sample within IDDP + Habanero reported ranges
    wob = np.random.uniform(50, 220, n_samples)  # IDDP: 50–220 kN
    rpm = np.random.uniform(60, 180, n_samples)  # IDDP: 60–180 RPM
    flow = np.random.uniform(30, 60, n_samples)  # Mud flow (IDDP logs)

    # Temperature: IDDP-2 reached 426°C at 4.6 km
    temp = 80 + (426 - 80) * (depth / 4600)
    temp = np.clip(temp, 80, 426)

    # Base ROP by lithology (Habanero & IDDP reports)
    base_rop = np.where(
        lithology == 1,  # granite → slow
        np.random.uniform(0.8, 4.0, n_samples),
        np.random.uniform(2.0, 12.0, n_samples)  # basalt/rhyolite
    )

    # Bit wear model (Habanero: bit life ~15–20 hrs in granite)
    bit_wear = np.ones(n_samples)
    for i in range(1, n_samples):
        # Accelerated wear in granite (lith=1) and high WOB/RPM
        wear_rate = 0.0003 * (wob[i] / 220 + rpm[i] / 180)
        if lithology[i] == 1:
            wear_rate *= 1.8  # granite is more abrasive
        bit_wear[i] = max(0.3, bit_wear[i-1] - wear_rate)

    # Apply bit wear to ROP
    rop = base_rop * bit_wear

    # Temperature penalty (ROP drops above 250°C — IDDP-2)
    temp_penalty = np.clip((temp - 250) / 150, 0, 1)
    rop = rop * (1 - 0.6 * temp_penalty)

    # Torque: baseline + spikes in fractured zones (Habanero dysfunction logs)
    torque = 5 + 0.02 * wob + 0.015 * rpm  # base torque
    for i in range(n_samples):
        # Simulate stick-slip in transitions or granite
        if (i > 0 and lithology[i] != lithology[i-1]) or lithology[i] == 1:
            if np.random.rand() < 0.12:  # 12% chance of spike
                torque[i] += np.random.uniform(3, 8)  # torque spike

    # Final clipping to Table 1 ranges
    wob = np.clip(wob, 20, 250)
    rpm = np.clip(rpm, 40, 200)
    flow = np.clip(flow, 10, 70)
    torque = np.clip(torque, 1, 18)
    temp = np.clip(temp, 80, 350)
    rop = np.clip(rop, 0.5, 18.0)

    return pd.DataFrame({
        'Depth': depth,
        'Lithology': lithology,  # 0=basalt, 1=granite, 2=rhyolite, 3=andesite
        'WOB': wob,
        'RPM': rpm,
        'Flow': flow,
        'Torque': torque,
        'Temp': temp,
        'ROP': rop
    })

# Generate and save
data = generate_realistic_iddp_habanero_data(1000)
data.to_csv("geothermal_realistic_1000.csv", index=False)
print("✅ Saved 'geothermal_realistic_1000.csv'")
print(f"📊 ROP range: {data['ROP'].min():.2f}–{data['ROP'].max():.2f} m/hr")
print(f"🌡️ Temp range: {data['Temp'].min():.1f}–{data['Temp'].max():.1f}°C")
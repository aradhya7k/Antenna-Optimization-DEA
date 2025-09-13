# 🛰️ Antenna Design and Optimization using PyAEDT & Differential Evolution  

## 📌 Project Overview  
This project focuses on the **design and optimization of coax-fed and inset-fed microstrip patch antennas**, as well as antenna arrays, operating at **5.65 GHz**. The workflow integrates **HFSS (Ansys Electronics Desktop)** with **Python scripting (PyAEDT)** to automate simulation and optimization.  

Key contributions of this project include:  
- Automating antenna modeling in **HFSS** using **PyAEDT**.  
- Implementing **Differential Evolution (DE)** for parameter optimization.  
- Optimizing **antenna parameters** (patch dimensions, feed positions, substrate size, inter-element spacing, etc.).  
- Targeting improved **S-parameters** (S11, S22, … Snn), **gain**, and **broadside radiation**.  
- Building a foundation for **AI/ML-assisted surrogate modeling** for faster optimization.  

---

## ⚙️ Project Workflow  

### 1. Antenna Modeling  
- Coax-fed and inset-fed **microstrip patch antennas** designed in HFSS.  
- **2×2 and 8-element arrays** analyzed for performance improvements.  
- Parameters considered:  
  - Patch length & width  
  - Inset gap & inset distance  
  - Feed width  
  - Substrate length & width  
  - Inter-element distance (for arrays)  

### 2. Optimization with Differential Evolution (DE)  
- Implemented full **DE loop in Python**.  
- Automated parameter updates, simulation runs, and result extraction via **PyAEDT**.  
- Fitness function defined based on:  
  - **S11 minimization** at 5.65 GHz (return loss improvement).  
  - **Broadside gain enhancement** for arrays.  
  - Penalty functions applied for frequency deviation.  

### 3. Results & Analysis  
- Improved **S11** (< -10 dB across target frequency).  
- Optimized **broadside gain** in array configurations.  
- Validation of results using HFSS plots:  
  - **Return Loss (dB)**  
  - **Radiation Patterns**  
  - **Gain vs Frequency**  

---

## 📊 Tools & Technologies  
- **HFSS (ANSYS Electronics Desktop)** – Antenna simulation  
- **PyAEDT** – Python scripting interface for HFSS  
- **Python (NumPy, SciPy, Matplotlib)** – Optimization & data handling  
- **Differential Evolution (DE)** – Optimization algorithm  

---

## 🚀 Applications  
- **5G & Wi-Fi communications** (5.65 GHz band)  
- **IoT and UAV systems**  
- **Aerospace & defense communication systems**  
- **Smart antenna arrays** for beamforming  

---

## 🔮 Future Work  
- Integration of **surrogate models (ML/AI)** for faster optimization.  
- Multi-objective optimization (S11, Gain, Bandwidth, Efficiency).   
- GUI for user-friendly antenna parameter control.  

---

## 👨‍💻 Contributors  
- **Shreyas Aradhya K** – Electronics Engineer    

---

## 📜 License  
This project is for **academic and research purposes**. For commercial use, please contact the author.  

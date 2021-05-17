# README

This ***prosumager*** model is developed for ***NewTRENDs Project*** and ***master thesis of Thomas***.

Hey, check this out: [Prosumager](https://songminyu.github.io/Prosumager/).



### 1 Milestones

- 01-06-2021: ready to run optimization
- 01-07-2021: finalize the optimization (code and database) and summarize the results

### 2 Next steps

#### 2.1 Songmin

- generate the hot water demand profile table
- calculate the energy demand for hot water

#### 2.2 Thomas

- try some optimization examples
- dynamic COP: update it in the database: updated COP_air and COP_water
- communicate with Philipp and start trying to set up the optimization model
- Update parameters of databank: COP, PV_BaseProfile, ID_PVType (kWp)

#### 2.3 Philipp

- work on the "B3\_Building.py"
- update building parameter tables --> ID_BuildingOption table
- write to Giacomo for a profile with heating and cooling for validation

### 5 Questions to discuss

#### 5.1 Space heating and hot water modeling

- Both systems share one boiler, and we only consider heat pump boiler to reduce the optimization cases.

- Both systems share one tank. 

  > (1) We assume water at 45 degree are taken from the tank for hot water demand. We still assume this for summer: the big tank is heated up to 45 degree. Since this temperature is not too high, the heat loss is limited.
  >
  > (2) We assume the energy heating water from 10 to 45 is an extra part provided by the boiler, only for hot water demand. So, this is not modeled as "loss of tank" for space heating. This doesn't influence the modeling and optimization of space heating at all. 

- For hot water demand, the water is further heated up from 45 to 65 (assumption) by an "electric heater". Here, we assume the electric heater is also the same heat pump (boiler). Its energy output should satisfy the hourly demand profile generated based on HOTMAP. Based on HOTMAP, we generate the 8760-hour hot water demand profile for one person by assuming: (1) daily consumption is 60kg; (2) the electric heater heat the water up for 20 degrees.

- In summary

  > (1) We optimize for space heating demand, but do not optimize for hot water demand.
  >
  > (2) The total energy consumption for hot water is calculated in two parts: first, heating water from 10 to 45 by the "shared boiler"; second, heating water from 45 to 65 by the "electric heater".
  >
  > (3) To reduce the cases for optimization, we only consider heat pump for both "shared boiler" and "electric heater". After the optimization, we can generate energy consumption for other options for the "shared boiler" and "electric heater", for example, a gas boiler and a regular electric heater. But, this is faster than running the optimization.

#### 5.2 Electricity load profile

- The electricity load profile excludes HotWaterDemand and SpaceHeating, but includes SmartAppliances (dryer, dishwasher, waschingmachine). The profile from Giacommo is about 2376 kWh. The research shows 4800 kWh (Kandler, 3 persons), 4344 kWh (Klingler, mean SFH), 3800 kWh (Fischer, 3 person). 
- So may we will use the mean value of 4344 kWh. The profile from Giacommo includes the household devices (SmartAppliances), so we have to subtract this (about 600 kWh in our calculation). For this we get 3744 kWh, but we includes the error, that we can subtract the Appliances hourly fitting, only on a 24 hour average. 
- In the future we can insert more electricity profiles in the database and apply an sensitivity analyses in our model. Also we may can show differences in the economical savings by selecting different profiles (different in yearly demand, but also different in temporal demand).

#### 5.3 Only optimize for typical days or weeks

- generate the base electricity demand profile for representative households on typical days
- we only optimize for the typical days, but they need to be selected carefully
- based on the results of these typical days or weeks, we generate the 8760-hour operation profile of the household


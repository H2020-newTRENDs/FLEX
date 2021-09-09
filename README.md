# README

This ***prosumager*** model is developed for ***NewTRENDs Project*** and ***master thesis of Thomas***.

Hey, check this out: [Prosumager](https://songminyu.github.io/Prosumager/).



### 1 Final output

- code opensource
- documentation on readthedoc
- web-based frontend

### 2 Next steps
- try out pandas.io.sql for faster data tranfer from sqlite databse (may not be better)
- create the DHW profile in the C3_tablegenerator (currently the profile is importet (thomas profile))
- find alternative profiles for DHW and electricity for single households (can profile generator generate many profiles 
simultaneously)
- integrate other heating systems (no HP) and get Water source Heat pump to work


#### 2.1 Songmin

- finalize the code

#### 2.2 Thomas

- write the thesis

#### 2.3 Philipp

- validation data from colleague (Energy plus)
- weather data

### 3 Reference scenario

#### Research Question - 1

- Model setup

#### Research Question - 2

- S1: base load, smart appliance (constant starting hours), space heating (heat pump), space cooling, hot water, PV --- without optimization, constant electricity price
- S2: S1 + Battery --- optimization, constant electricity price

#### Research Question - 3

- S2: S1 + Battery --- optimization, constant electricity price
- S3: S1 + Battery --- optimization, variable electricity price

#### Research Question - 4

- S3: S1 + Battery --- optimization, variable electricity price
- S4: S1 + Battery + EV --- optimization, variable electricity price



**Chapter: Application of the Model: Research Question 3 and 4**







### 5 Questions to discuss

#### 5.1 Space heating and hot water modeling

- Both systems share one boiler, and we only consider heat pump boiler to reduce the optimization cases.

- Both systems share one tank. 

  > (1) We assume water at 45 degree are taken from the tank for hot water demand. We still assume this for summer time: the big tank is heated up to 45 degree. Since this temperature is not too high, the heat loss is limited.
  >
  > (2) To cut between the modeling of space heating and hot water, we assume there is some energy provided by the same boiler to heat water from 10 to 45 degree and specifically for hot water demand. So, this part doesn't influence the modeling and optimization of space heating at all. It is not even an "exogenous heat loss", either. 
  >
  > (3) After being taken from the tank at 45 degree, the water is further heated up from 45 to 65 by an "electric heater". Here, we assume the electric heater is also the same heat pump (boiler). This "second part" energy output should satisfy the hourly demand profile generated based on HOTMAP. Based on HOTMAP, we generate the 8760-hour hot water demand profile for one person by assuming: (1) daily consumption is 60kg; (2) the electric heater heat the water up for 20 degrees.

- For three reasons, we do not (directly) optimize for hot water demand. 

  > (1) "Giving hot water system a separate tank" is not a realistic assumption, and there are too many ways in reality about how "hot water system" share storage with the "space heating system". 
  >
  > (2) It can only be (directly) optimized when it has its own storage. 
  >
  > (3) When there is a battery, the hot water system can be indirectly optimized. 

- In summary

  > (1) To reduce the cases for optimization, we only consider heat pump for both "shared boiler" and "electric heater". After the optimization, we can generate energy consumption for other options for the "shared boiler" and "electric heater", for example, a gas boiler and a regular electric heater. But, this is faster than running the optimization.
  >
  > (2) The total energy consumption for hot water is calculated in two parts: first, heating water from 10 to 45 by the "shared boiler"; second, heating water from 45 to 65 by the "electric heater".
  >
  > (3) The "first part" is provided by the "boiler", and we can calculate its amount. If the boiler is heat pump and consumes electricity, the consumption is **allocate it to the same hours when the "electric heater (second part)" is used**. 

#### 5.2 Electricity load profile

- The electricity load profile excludes HotWaterDemand and SpaceHeating, but includes SmartAppliances (dryer, dishwasher, waschingmachine). The profile from Giacommo is about 2376 kWh. The research shows 4800 kWh (Kandler, 3 persons), 4344 kWh (Klingler, mean SFH), 3800 kWh (Fischer, 3 person). 
- So may we will use the mean value of 4344 kWh. The profile from Giacommo includes the household devices (SmartAppliances), so we have to subtract this (about 600 kWh in our calculation). For this we get 3744 kWh, but we includes the error, that we can subtract the Appliances hourly fitting, only on a 24 hour average. 
- In the future we can insert more electricity profiles in the database and apply an sensitivity analyses in our model. Also we may can show differences in the economical savings by selecting different profiles (different in yearly demand, but also different in temporal demand).

#### 5.3 Reference point without optimization

To calculate a reference "operation cost" for the overall system, we need to swithing off the optimization for all the aspects.

> 1. Appliance: we have the property for this
> 2. PV, Battery and EV: options with zero area or capacity, which also represents the "not adopted" situation
> 3. Space heating and cooling: set extra constraint on the tank and room temperature and set it constant

Based on this reference point, we can add all the flexibility one by one and show the contribution to reduce the overall cost.

We can ask Giacomo for data to validate model without optimization.

#### 5.4 Building standard and size of heat pump

The hourly optimization of space heating requires a minimum power (boundary) of the heat pump to satisfy the heat demand. For the worst building case this is about 20 kW (thermal power). To satisfy the heat demand of a low energy building with this 20 kW brings unrealistic results, because the heat pump will run on 20 kW while the best situation (Photovoltaic or Price) and 20 kW are totally oversized for a low energy building 

- toDo: Include a boundary of the maximum HP Power in the database, connected to the building standard 

#### 5.5 Database Solar radiation
The solar radiation from different cilestial directions is calculated for germany only at the time being. 
(with longitude and altitude = Würzburg) How can we integrate this into the SQLite database?

- The temperature of each country is the average of all 10x10 km regions

#### 5.6 V2B and Driving profiles 
The bidirectional charge / discharge of the EV offers the possibility to use the EV as a battery. 

- V2B: Discharge the EV to supply Loads or charge the EV
- At the moment the car is every day from 7-17 away. Special profiles for saturday and sunday offers a potential for V2B

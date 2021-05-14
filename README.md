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
- dynamic COP: update it in the database
- communicate with Philipp and start trying to set up the optimization model

#### 2.3 Philipp

- work on the "B3\_Building.py"
- update building parameter tables --> ID_BuildingOption table
- write to Giacomo for a profile with heating and cooling for validation

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

#### 5.2 Only optimize for typical days or weeks
- generate the base electricity demand profile for representative households on typical days
- we only optimize for the typical days, but they need to be selected carefully
- based on the results of these typical days or weeks, we generate the 8760-hour operation profile of the household

#### 5.3 Reference point without optimization

- To calculate a reference "operation cost" for the overall system, we need to swithing off the optimization for all the aspects.

  > - Appliance: we have the property for this
  > - PV, Battery and EV: options with zero area or capacity, which also represents the "not adopted" situation
  > - Space heating and cooling: set extra constraint on the tank and room temperature and set it constant

- Based on this reference point, we can add all the flexibility one by one and show the contribution to reduce the overall cost.

- We can ask Giacomo for data to validate model without optimization.
























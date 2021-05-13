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

#### 5.1 Model optimization coverage

- smart appliance: dryer, dish washer, washing machine
- space heating
- space cooling
- PV
- battery
- EV

#### 5.2 Hot water modeling

- the 8760h demand profile is calcuated for 1 person based on HOTMAP

- no specific tank for hot water, no optimization. The energy consumption is calculted in two parts: 

  > (1) from 10 to 45, based on either heat pump or other boiler of the space heating system --> water is taken from space heating tank, even in summer, because the tank in summer is only 45 degree, so even though we heat up a large tank only for hot water, the heat loss is limited (we assume). 
  >
  > (2) from 45 to 65, based on heat pump.

#### 5.3 Space heating modeling

- only consider heat pump boiler to reduce the optimization cases
- interaction with hot water: the heat pump boiler of space heating provide the energy heating water from 10 to 45, separately. So, even though the space heating and hot water share one same tank, the hot water doesn't influence the modeling and optimization of space heating system.

#### 5.4 Only optimize for typical days or weeks
- generate the base electricity demand profile for representative households on typical days
- we only optimize for the typical days, but they need to be selected carefully
- based on the results of these typical days or weeks, we generate the 8760-hour operation profile of the household



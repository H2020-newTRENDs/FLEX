# README

This ***prosumager*** model is developed for ***NewTRENDs Project*** and ***master thesis of Thomas***.

Hey, check this out: [Prosumager](https://songminyu.github.io/Prosumager/).



### 1 Milestones

- 01-06-2021: ready to run optimization
- 01-07-2021: finalize the optimization (code and database) and summarize the results

### 2 Next steps

#### 2.1 Songmin

- keep working on the files in "B\_Classes" and "C1\_TableGenerator"
- Set up one example for start and let Philipp know, to set up the one for building class.
- update the database

#### 2.2 Thomas

- try some optimization examples
- literature review
- dynamic COP: pre-calculate a table like electricity price (see the excel). Songmin will set up the table structure and Thomas will fill in the numbers.

#### 2.3 Philipp

- get familiar with the code and database
- work on the "B3\_Building.py"
- maybe change the calculation with other parameters (-- please let Songmin know which parameters are used)

### 5 Questions to discuss

#### 5.1 Literature review

- Summarize the aspects that are optimized in our model and compare it with existing studies

  > - smart appliance
  > - space heating
  > - space cooling?
  > - hot water?
  > - PV
  > - battery
  > - EV

- Apart from covering more aspects in the optimization, are there other contributions?

  > - data resolution?
  > - geographic coverage?

#### 5.2 Define the functions

- have a look at the updated classes and database
- go over the optimization process together and define functions in each class

#### 5.3 Hot water

- Demand profile

  > - From INVERT, we have demand in kWh/m2. Then, we find **average persons living in each building type** and translate it to kWh/m2 for each person. Then, based on HOTMAP data, we allocate the annual demand to hourly profiles, and at the same time, we have distinction between working days and holidays.
  > - **Songmin**: I briefly compared relevant parameters from three sources - (1) INVERT; (2) calculation by my colleague at ISI; (3) 60kg per day. We can have a look together in the next meeting.

- Optimization

  > - Water is taken from same tank of space heating, then heated up for higher temperature with electric heater. Then, the energy comsumer by the electric heater is decided by the tank temperature. To simplify the optimization, we can assume that the water feed to the electric heater at a constant temperature. Then, from the space heating optimization perspective, the energy goes to hot water is an exogenous heat loss of the tank.
  >
  > - For now, I haven't removed the tank in HotWater. **A few questions:**
  >
  >   > - How about in summer, when the space heating system is turned off? 
  >   > - In the FORECAST mode, I also see oil/gas/coal/biomass boiler for hot water (consumption of these energy carriers). Does this come from the same boiler for space heating? Or, the are used same as the "electric heater" mentioned above? Besides, I also see heat pump for hot water. 


#### 5.4 Building parameter

- when Philipp decides what calculation method to use, we will know which parameters are relevant and to be collected.

#### 5.5 PyCharm tips

- change names for all




# README

This ***prosumager*** model is developed for ***NewTRENDs Project*** and ***master thesis of Thomas***.



### 1 Milestones

- set up the code and database by the end of May (i.e. ready to run the optimization)

### 2 Next steps

#### 2.1 Songmin

- keep working on the files in "B\_Classes" and "C1\_TableGenerator"
- update the database
- prepare table templates for Philipp

#### 2.2 Thomas

- get familiar with the code and database
- try some optimization examples

#### 2.3 Philipp

- get familiar with the code and database
- send Songmin the tables
- work on the "B3\_Building.py"

### 3 Coding Conventions

#### 3.1 Naming convention

- ***variable***: **V**ariable**N**ame
- ***parameter defined in a function***: **p**arameter_**n**ame
- ***function***: **calc_B**uilding**S**pace**H**eating**C**ost
- ***table***: register in the REG, then use it in the code

#### 3.2 Commiting to Github

- First, pull when you start working and make sure you are working on the latest version.
- Second, pull again before pushing, because during your working time, the others may have pushed. However, even though that is the case, the revisions should be merged into your local version automatically, as long as we didn't work on the same row. If you do see the conflicts, solve it locally. ***Always pull before pushing, so that we solve the conflicts locally.***
- Third, after solving the conflict, pull again (it should be "already up to date"), and push super-fast! Hahaha

#### 3.3 Other conventions

- If you need something from the "Infrastructure", tell Songmin to create them.

### 4 Git Commands

- git pull origin master
- git add .
- git commit -m"***comments for this commit***"
- git push origin master

### 5 Questions

#### 5.1 Hot water profile

For hot water, the optimization constraint is to satisfy a "hot water demand profile" in each hour. The profile should be in the unit of "kWh". This is different from the space heating or cooling, which are to satisfy a "temperature profile" in the unit of degree. 

However, I am not sure how to generate the hourly profile yet. I think we have two ways:

First is the way you did it last time, Philipp. Hot did you do that?

Second is HOTMAP method. I have tried it. It can work. Basically, there are two steps:

**1. Estimate the total annual hot water useful energy demand profile (unit: kWh).**

- Philipp, I see you have "DHW_per_day" in your "User_profiles_example" table. Is it in the unit of kWh? 
- TABULA also provides estimation for hot water demand, but it's for different building type and age classes. Not sure how this is related to number of persons in the household.
- Thomas also found a number: for each person, daily hot water demand is 60kg. But we need to translate it to "kWh" first. But this might be complicated because we need to consider the temperature difference between "ground water" and "target temperature" in the hot water tank. This is different in different seasons.

> Update: 
>
> Since we have no reliable "temperature data" for hot water consumption, we cannot translate "kg" to "kWh". It seems that the best we have now is this "building dependent annual demand" for hot water from INVERT. Based on HOTMAP data, we allocate this annual number to 8760 hours in a year.
>
> On the other hand, in the model we can optimize
>
> - "energy from tank to hot water use" in the unit of kWh
> - "energy from boiler to tank" in the unit of kWh
>
> But, it seems we still need a temperature variable in the tank, to calculate the hourly heat loss. We may assume 50 degree as bottom limit if this parameter is not too sensitive?
>
> Things not solved:
>
> - still, we need to somehow relate the annual hot water demand (kWh) to person instead of living area (m2). Then, we can further introduce "lifestyle assumptions (WFH days)" to the analysis.
> - one parameter can help: average persons living in each building type.

**2. Allocate the annual demand to 8760 hours.**

- In HOTMAP, I think there is lifestyle assumption embedded in the empirical data they used. We may need to have a look at it and see if it is aligned with our scenario assumptions of lifestyle. But this is not a big deal since the total demand of hot water is relative less compared with space heating. We can check it and maybe change it later after Thomas's master thesis.

#### 5.2 Base year for all?

I have radiation and temperature for 2010 and we will use it as base year. But, I am not sure if this one year data is representative? It is real data, but do we need to use more years data to run the optimization? Or, we somehow calibrate a "representative year data"? 

This is also not in hurry. We could discuss about it later.

#### 5.3 Building parameter

- The meaning of "heat transmission parameter"? Conductivity?
- Compare the parameter in the calculation table from Mahsa.
- Each row represents a combination of building components? Then, renovation is modeled as switching between rows? Where can we find detailed information of components (impacts and cost)? IWU database?














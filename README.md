# README

# Table of contents
1. [FLEX MODEL](#FLEX_MODEL)
2. [Getting started](#Getting_started)
   1. [FLEX Operation](#FLEX_Operation)
   2. [FLEX Community](#FLEX_Community)
   3. [FLEX Investment](#FLEX_Investment)


# FLEX Model <div id="FLEX_MODEL"/>
Status: The mode is still under development
## Overview and scope
This model was created to calculate the energy demand for single buildings
considering consumers, prosumers and prosumagers (a prosumager is a household
which consumes, produces and manages it own energy). The aim is to calculate
the influence of single technology adoptions for individual buildings and 
determine the difference on energy demand when switching from a prosumer to a
prosumager. In this model consumer and prosumer are defined as households who 
do not adapt their energy consumption behavior to any outside influence. Indoor
temperature comfort is kept throughout the year. A prosumager on the other hand
rationally tries to minimize the overall operation cost by maximizing PV self-consumption
etc. 

### Key features
- Hourly optimization of electric appliances in a building
- considering PV, Battery, thermal storages, thermal mass of the building, heat pumps, electric vehicles, direct 
   electric heater, air conditioner

## Input data
The input data is provided in the form of Excel sheets in data/input_operation.

## Results
Results are generated in sqlite files. 

### Explanation of Result Variables
Results are always divided into "reference" and "optimization" results representing the same scenarios for consumers
and prosumers versus prosumagers respectively. 
#### Yearly Results
Yearly results represent show the total sum of each variable over the year, like the energy cost in total or the 
total amount of space heating provided.
#### Hourly Results
The tables with hourly results contain vectors of length 8760 for each variable in each scenario. 

# Getting started <div id="Getting_started"/>

### 

## FLEX Operation <div id="FLEX_Operation"/>


## FLEX Community <div id="FLEX_Community"/>


## FLEX Investment <div id="FLEX_Investment"/>





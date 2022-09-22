from enum import Enum


class InterfaceTable(Enum):

    # from operation to investment
    OperationEnergyCost = "InvestmentScenario_OperationEnergyCost"
    OperationEnergyCostChange = "InvestmentScenario_OperationEnergyCostChange"


class Color(Enum):
    Appliance = "tab:orange"
    SpaceHeating = "tab:red"
    HotWater = "tab:green"
    SpaceCooling = "tab:blue"
    BatteryCharge = "tab:purple"
    VehicleCharge = "tab:brown"
    Grid = "tab:gray"
    PV = "gold"
    BatteryDischarge = "tab:olive"
    VehicleDischarge = "tab:cyan"
    PV2Grid = "tab:pink"
    GridConsumption = "tab:brown"
    PVConsumption = "gold"
    GridFeed = "tab:green"
    # DIMGRAY = "dimgray"
    # LIGHTCORAL = "lightcoral"
    # TOMATO = "tomato"
    # PERU = "peru"
    # DARKORANGE = "darkorange"
    # GOLD = "gold"
    # OLIVEDRAB = "olivedrab"
    # LIME = "lime"
    # DARKSLATEGREY = "darkslategrey"
    # ROYALBLUE = "royalblue"
    # VIOLET = "violet"
    # NAVY = "navy"
    # CRIMSON = "crimson"

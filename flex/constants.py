from enum import Enum
















class InterfaceTable(Enum):
    value: str

    # from operation to investment
    OperationEnergyCost = "InvestmentScenario_OperationEnergyCost"
    OperationEnergyCostChange = "InvestmentScenario_OperationEnergyCostChange"


class Color(Enum):
    value: str

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
    P2P_Profit = "tab:brown"
    Opt_Profit = "tab:olive"
    P2P_trading = "tab:brown"
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

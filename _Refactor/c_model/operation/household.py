
from _Refactor.b_basic.household.household import Household

class OperationHousehold(Household):

    def test(self):
        print("f_test")

    def setup(self):
        self.region.setup()
        self.person_list.setup()
        self.building.setup()
        self.appliance_list.setup()
        self.boiler.setup()
        self.tank.setup()
        self.air_conditioner.setup()
        self.pv.setup()
        self.battery.setup()
        self.vehicle.setup()
        self.behavior.setup(self)
        self.demand.setup(self)
        self.policy.setup(self)

if __name__ == "__main__":
    h = OperationHousehold()
    h.test()



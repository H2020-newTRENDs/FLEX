
from core.elements.component import Component


class FeedInTariff(Component):

    def __init__(self, component_id):
        self.ID_FeedInTariff: int = None
        self.name: str = None
        self.feed_in_tariff = None
        self.unit = None

        self.set_parameters(component_id)


class Component:

    def set_params(self, params: dict):
        for param_name, param_value in params.items():
            if param_name in self.__dict__.keys():
                setattr(self, param_name, param_value)


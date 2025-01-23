# from .MPC import MPC
# from .MPC2 import MPC2
# from .MPC2Pro import MPC2Pro
# from .MPC3 import MPC3
# from .Policy import Policy
import importlib

def get_item(name):
    # dict = {
    #     "MPC": MPC,
    #     "MPC2": MPC2,
    #     "MPC3": MPC3,
    #     "MPC2Pro": MPC2Pro,
    #     "Policy": Policy
    # }
    module = importlib.import_module("mymbrl.controllers."+name)
    module_class = getattr(module, name)
    return module_class
    # return dict[name]
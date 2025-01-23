import importlib

def get_item(name):

    module = importlib.import_module("mymbrl.agents."+name)
    module_class = getattr(module, "AgentItem")
    return module_class
    # return dict[name]
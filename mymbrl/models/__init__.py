import importlib

def get_item(name):
    module = importlib.import_module("mymbrl.models."+name)
    module_class = getattr(module, "DynamicModel")
    return module_class
    
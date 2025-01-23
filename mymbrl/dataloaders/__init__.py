import importlib

def get_item(name):
    module = importlib.import_module("mymbrl.dataloaders."+name)
    module_class = getattr(module, 'MyDataloader')
    return module_class
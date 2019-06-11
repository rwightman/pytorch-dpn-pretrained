""" PyTorch DPN hubconf.py

## Users can get this published model by calling:
hub_model = hub.load(
    'rwightman/pytorch-dpn-pretrained:master', # repo_owner/repo_name:branch
    'dpn92', # entrypoint
    pretrained=True) # kwargs for callable
"""
dependencies = ['torch', 'math']

from dpn import dpn68, dpn68b, dpn92, dpn98, dpn131, dpn107

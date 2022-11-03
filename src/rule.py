
import rhinoinside
rhinoinside.load()
import System
import Rhino
import Rhino.Geometry as rg
import random

# The rule here indicates the next port you want to connect to!
# If you have maximum 4 rules, the bigget number you can enter then is 3

class Rule():
    def __init__(self, num_sels, option):
        self.num_sels = num_sels
        self.option = option
        random.seed(option)
    
    def get_rules(self):
        rules = [
            'top',
            'left',
            'right',
            'bottom'
        ]
        return rules
    
    def get_rule_seq(self):
        sel_rules_idxs = []
        for n in range(self.num_sels):
            sel_rules_idxs.append(random.choice(range(0,4)))
        return sel_rules_idxs
    






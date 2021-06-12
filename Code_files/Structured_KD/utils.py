import logging

def print_model_parm_nums(model, string):
    b = []
    for param in model.parameters():
        b.append(param.numel())
    print(string + ': Number of params: %.2fM', sum(b) / 1e6)
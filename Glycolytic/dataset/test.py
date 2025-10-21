import numpy as np
from pysindy.feature_library import PolynomialLibrary
from pysindy import SINDy
import pysindy as ps

import math
np.math = math

file_name = 'pendulum'
train = np.load('glycolytic_evo.npy')
print(train.shape)
train_list = [train[i] for i in range(train.shape[0])] 
# time = np.arange(0, 50 + 0.05, 0.05)  # 从 0 到 50，步长为 0.05
# print(time.shape)  # 应该输出 (1001,)
opt = ps.STLSQ(threshold=0.08, alpha=0.5)
poly = PolynomialLibrary(degree=2)
model = SINDy(optimizer=opt, feature_library=poly)
model.fit(train_list, t=0.001, multiple_trajectories=True)
model.print()

from sympy import symbols
import re
def export_sindy_as_single_torch_function_keep_vars(model, tol=1e-10):
    """
    生成单一函数 f(x, device=None)，返回 torch.tensor([...])
    变量名保持原样，不替换成 x[1]，但补乘号和幂运算符。
    """
    coef_matrix = model.coefficients()
    feature_names = model.get_feature_names()

    exprs = []
    for coefs in coef_matrix:
        terms = []
        for c, name in zip(coefs, feature_names):
            if abs(c) > tol:
                if name == '1':
                    terms.append(f"{c:.5f}")
                else:
                    # 用正则把 ^ 替换为 **
                    name_fixed = re.sub(r'\^', r'**', name)
                    # 用乘号替换变量间空格，比如 "x1 x5" -> "x1*x5"
                    name_fixed = '*'.join(name_fixed.split())
                    terms.append(f"{c:.12f} * {name_fixed}")
        expr = " + ".join(terms) if terms else "0.0"
        exprs.append(expr)

    joined_exprs = ",\n        ".join(exprs)
    func_str = f"""
    return torch.cat([
        {joined_exprs}
    ], dim=-1)
"""
    return func_str


print(export_sindy_as_single_torch_function_keep_vars(model))
#!/usr/bin/env python
# Created by "Thieu" at 09:57, 17/06/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import statsmodels.datasets.co2 as co2
from statsmodels.tsa.seasonal import seasonal_decompose

dataset = co2.load(as_pandas=True).data
print(dataset)
dataset = dataset.fillna(dataset.interpolate())
data = seasonal_decompose(dataset)
print(data.resid)


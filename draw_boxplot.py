#!/usr/bin/env python
# Created by "Thieu" at 22:03, 02/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

# import the required library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mealpy.evolutionary_based import DE, CRO, GA
from mealpy.math_based import CGO, GBO, PSS
from mealpy.bio_based import SMA, EOA, BBO
from mealpy.system_based import AEO, GCO
from mealpy.human_based import TLO, FBIO, GSKA, QSA
from mealpy.music_based import HS
from mealpy.physics_based import HGSO, NRO, EFO, EO
from mealpy.swarm_based import DO, GWO, HGS, HHO, WOA, PSO, SSA, PFA


## Define models
epoch = 500
pop_size = 50

model1 = DE.BaseDE(epoch=epoch, pop_size=pop_size)
model2 = CRO.OriginalCRO(epoch=epoch, pop_size=pop_size)
model3 = GA.BaseGA(epoch=epoch, pop_size=pop_size)
model4 = CGO.OriginalCGO(epoch=epoch, pop_size=pop_size)
model5 = GBO.OriginalGBO(epoch=epoch, pop_size=pop_size)
model6 = PSS.OriginalPSS(epoch=epoch, pop_size=pop_size)
model7 = SMA.BaseSMA(epoch=epoch, pop_size=pop_size)
model8 = EOA.OriginalEOA(epoch=epoch, pop_size=pop_size)
model9 = BBO.BaseBBO(epoch=epoch, pop_size=pop_size)
model10 = AEO.OriginalAEO(epoch=epoch, pop_size=pop_size)
model11 = GCO.BaseGCO(epoch=epoch, pop_size=pop_size)
model12 = TLO.BaseTLO(epoch=epoch, pop_size=pop_size)
model13 = FBIO.OriginalFBIO(epoch=epoch, pop_size=pop_size)
model14 = GSKA.BaseGSKA(epoch=epoch, pop_size=pop_size)
model15 = QSA.OriginalQSA(epoch=epoch, pop_size=pop_size)
model16 = HS.BaseHS(epoch=epoch, pop_size=pop_size)
model17 = HGSO.OriginalHGSO(epoch=epoch, pop_size=pop_size)
model18 = NRO.OriginalNRO(epoch=epoch, pop_size=pop_size)
model19 = EFO.OriginalEFO(epoch=epoch, pop_size=pop_size)
model20 = EO.OriginalEO(epoch=epoch, pop_size=pop_size)
model21 = DO.OriginalDO(epoch=epoch, pop_size=pop_size)
model22 = GWO.OriginalGWO(epoch=epoch, pop_size=pop_size)
model23 = HGS.OriginalHGS(epoch=epoch, pop_size=pop_size)
model24 = HHO.OriginalHHO(epoch=epoch, pop_size=pop_size)
model25 = WOA.OriginalWOA(epoch=epoch, pop_size=pop_size)
model26 = PSO.OriginalPSO(epoch=epoch, pop_size=pop_size)
model27 = SSA.OriginalSSA(epoch=epoch, pop_size=pop_size)
model28 = PFA.OriginalPFA(epoch=epoch, pop_size=pop_size)

list_names = ["F12017", "F22017", "F32017", "F42017", "F52017", "F62017", "F72017", "F82017", "F92017", "F102017",
              "F112017", "F122017", "F132017", "F142017", "F152017", "F162017", "F172017", "F182017", "F192017", "F202017"]

list_models = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10,
               model11, model12, model13, model14, model15, model16, model17, model18, model19,
               model20, model21, model22, model23, model24, model25, model26, model27, model28]


# for idx_name, func_name in enumerate(list_names):
#     df_fit = {}
#     for idx_model, model in enumerate(list_models):
#         model_name = model.__class__.__name__
#         df = pd.read_csv(f"history/best_fit/{model_name}_best_fit.csv")
#         model_name2 = model_name.replace("Original", "").replace("Base", "")
#         df_fit[model_name2] = df.iloc[:, idx_name]
#     pd.DataFrame(df_fit).to_csv(f"history/best_fit_box_plot/{func_name}.csv")

cols = ["SSA", "HHO", "TLO", "EFO", "WOA", "GWO", "QSA", "GBO"]
for idx_name, func_name in enumerate(list_names):
    df = pd.read_csv(f"history/best_fit_box_plot/{func_name}.csv", index_col=0, usecols=cols)
    # df = df.sample(n=8, axis='columns')
    # df.boxplot(kind="box")
    sns.boxplot(x="variable", y="value", data=pd.melt(df))
    plt.savefig(f"history/best_fit_box_plot/{func_name}.png")
    plt.show()


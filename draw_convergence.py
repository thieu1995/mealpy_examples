#!/usr/bin/env python
# Created by "Thieu" at 12:22, 12/10/2022 ----------%                                                                               
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

list_names = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10",
              "F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19", "F20"]


def draw_all_trials():
    list_models = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10,
                   model11, model12, model13, model14, model15, model16, model17, model18, model19,
                   model20, model21, model22, model23, model24, model25, model26, model27, model28]
    titles = ["Trial 1", "Trial 2", "Trial 3", "Trial 4", "Trial 5", "Trial 6", "Trial 7", "Trial 8", "Trial 9", "Trial 10"]
    for idx_model, model in enumerate(list_models):
        for idx_name, func_name in enumerate(list_names):
            model_name = model.__class__.__name__

            df = pd.read_csv(f"history/convergence/{model_name}/{func_name}_convergence.csv", skiprows=1, names=titles)
            for idx_title, title in enumerate(titles):
                plt.plot(df.iloc[:, idx_title], label=title)

            plt.legend()
            plt.savefig(f"history/convergence/{model_name}/{func_name}.png")
            plt.show()


def draw_single_trial(trial="trial_1"):
    list_groups = [
        [model1, model2, model3, model8, model9, model26, model16],
        [model4, model5, model6, model7, model17, model20, model24],
        [model12, model13, model14, model15, model19, model23, model27],
        [model10, model11, model18, model21, model22, model25, model28]
    ]
    for idx_name, func_name in enumerate(list_names):
        for idx_group, groups in enumerate(list_groups):
            for idx_model, model in enumerate(groups):
                model_name = model.__class__.__name__
                model_title = model.__module__.split('.')[-1]
                df = pd.read_csv(f"history/convergence/{model_name}/{func_name}_convergence.csv")
                plt.plot(df.loc[:, trial], label=model_title)
            plt.title(f"{func_name}-2017")
            plt.xlabel("Iterations")
            plt.ylabel("Fitness value")
            plt.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(f"history/convergence/chart/{func_name}-{idx_group+1}.png")
            plt.show()


draw_single_trial()
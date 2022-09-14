#!/usr/bin/env python
# Created by "Thieu" at 22:36, 14/09/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

#### Using multiple algorithm to solve multiple problems with multiple trials

from opfunu.cec_based import cec2017
from mealpy.evolutionary_based import DE, CRO, GA
from mealpy.math_based import CGO, GBO, PSS
from mealpy.bio_based import SMA, EOA, BBO
from mealpy.system_based import AEO, GCO
from mealpy.human_based import TLO, FBIO, GSKA, QSA
from mealpy.music_based import HS
from mealpy.physics_based import HGSO, NRO, EFO, EO
from mealpy.swarm_based import DO, GWO, HGS, HHO, WOA, PSO, SSA, PFA
from mealpy.multitask import Multitask


def problem_generator(name, ndim):
    fobj = getattr(cec2017, name)(ndim, f_bias=0)
    problem = {
        "lb": fobj.lb.tolist(),
        "ub": fobj.ub.tolist(),
        "minmax": "min",
        "fit_func": fobj.evaluate,
        "name": fobj.name,
        "log_to": None,
    }
    return problem


list_names = ["F12017", "F22017", "F32017", "F42017", "F52017", "F62017", "F72017", "F82017", "F92017", "F102017",
              "F112017", "F122017", "F132017", "F142014", "F152017", "F162017", "F172017", "F182017", "F192017", "F202017"]
list_problems = [problem_generator(name, 30) for name in list_names]


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

list_models = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10,
               model11, model12, model13, model14, model15, model16, model17, model18, model19,
               model20, model21, model22, model23, model24, model25, model26, model27, model28]

## Define and run Multitask

if __name__ == "__main__":
    multitask = Multitask(algorithms=list_models, problems=list_problems)
    multitask.execute(n_trials=10, mode="parallel", n_workers=10, save_path="history", save_as="csv", save_convergence=True, verbose=True)

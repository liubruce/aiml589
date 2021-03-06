import numpy as np
from scipy.stats import t
#Peg
# Ant, halfcheetah, hopper, humanoid, humanoidstandup, reacher, swimmer, walker2d
x = np.array([
    [6160, 5810, 6050, 5960, 6120],
    [9310, 9230, 9250,9250,9230],
    [3470,3060,3510,1700,3510],
    [7200,7910,7150,7130,7600],
    [154000,156000,147000,155000,125000], [-12.1,-11.7,-12.5,-11.5,-10.7], [91.1,86.5,87.1,80.8,87.9],
    [5720,5690,5910,5640,5870]
]
)

# a = np.array([[1, 2], [3, 4]])
result = np.around(np.std(x, axis=1),decimals=0)

m = x.mean(axis=1)
s = x.std(axis=1)
dof = len(x)-1
confidence = 0.95
t_crit = np.abs(t.ppf((1-confidence)/2,dof))
print((m-s*t_crit/np.sqrt(len(x)), m+s*t_crit/np.sqrt(len(x))))
# print('PEG std are ', result)
#
# # ED2
# a = np.array([
#     [6040,5990,5500,6040,5700], [9300,9290,9290,9330,9150], [3420,3470,3200,1030,3470],[6190,7070,6750,7000,7020],
#     [141000,156000,156000,156000,155000], [-13.3,-14.2,-14.9,-15.2,-13.9], [89.7,87.5,86.6,93.6,85.9],
#     [4920,755,4910,5380,4810]]
# )
# result = np.std(a, axis=1)
# print('ED2 std are ', result)


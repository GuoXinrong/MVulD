import pandas as pd
import sys
from pathlib import Path

sys.path.append(str((Path(__file__).parent.parent)))

from scripts.eval_patches import BigVulDatasetDevignPatch
from models.devign.dataset import BigVulDatasetDevign
from utils import cache_dir
#
# dataset = 'bigvul'
# cache_path = cache_dir() / 'data' / dataset / f'{dataset}_cleaned.pkl'
# df = pd.read_pickle(cache_path)
# train_df = df[df.partition == 'train']
# valid_df = df[df.partition == 'valid']
# iiii = {176132, 178189, 178193, 178195, 176153, 176157, 178210, 176169, 176171, 180270, 178224, 178227, 176190, 176194,
#         178267, 178268, 176226, 176235, 176236, 178293, 178299, 176258, 176267, 182415, 176272, 178322, 176276, 178326,
#         176282, 178342, 176310, 178358, 176313, 178366, 176325, 178373, 176331, 178382, 176341, 176347, 178396, 176351,
#         178402, 176357, 178408, 180457, 176380, 178430, 176387, 178442, 178448, 176407, 178463, 178464, 178466, 174374,
#         182568, 178476, 174386, 180532, 182585, 176442, 182586, 182588, 182593, 176453, 182599, 180552, 182600, 180554,
#         180566, 176474, 176479, 178529, 176482, 176484, 178535, 178543, 176498, 178574, 174481, 174484, 176536, 182682,
#         178587, 176549, 176550, 178603, 178606, 178617, 178625, 176585, 178635, 176588, 178637, 174543, 182737, 174550,
#         174555, 178656, 176614, 178662, 178675, 182774, 178682, 178685, 176638, 174597, 182789, 182790, 182793, 178699,
#         182799, 178712, 182808, 180763, 182811, 176675, 182821, 178729, 178731, 174639, 176689, 176690, 182836, 178749,
#         174657, 174659, 182853, 174663, 174665, 174667, 176715, 176720, 176721, 174682, 174687, 178787, 174692, 178788,
#         176744, 174702, 182894, 178803, 178804, 176757, 178810, 174715, 182916, 174727, 174736, 182929, 176788, 174744,
#         176792, 178842, 174747, 176796, 174749, 182942, 174751, 182947, 174759, 178856, 176814, 176820, 176825, 178877,
#         178893, 176855, 174809, 176860, 178914, 176869, 174825, 178923, 176876, 178926, 174831, 178928, 176882, 176885,
#         174843, 176902, 183049, 174861, 183054, 178960, 174867, 174878, 183075, 176934, 178984, 176947, 176951, 179001,
#         181056, 174915, 176972, 179025, 174930, 179034, 179040, 174947, 176996, 179045, 174966, 179063, 181113, 174974,
#         174985, 181135, 181137, 177045, 179093, 175004, 175005, 177052, 175012, 175016, 181168, 177078, 183223, 175039,
#         183244, 177102, 175065, 175066, 183257, 183264, 183269, 179180, 183283, 175092, 177152, 175107, 175109, 175118,
#         183310, 183311, 177172, 177173, 183316, 183324, 183332, 175144, 177193, 177203, 183354, 183357, 175173, 175179,
#         175181, 177229, 175210, 177266, 179315, 179316, 175224, 183416, 177290, 175251, 177304, 177308, 175266, 175270,
#         177332, 175288, 175291, 177346, 175335, 175338, 175341, 175343, 175346, 175360, 177412, 177415, 177426, 181542,
#         175405, 175406, 177455, 177459, 177460, 175416, 177467, 177473, 175426, 175428, 179529, 175438, 179534, 177493,
#         175457, 175463, 179563, 177520, 177531, 177534, 177536, 177537, 175493, 177551, 177559, 177562, 175521, 179630,
#         175537, 183731, 179638, 179640, 179644, 183742, 177602, 177605, 179654, 183752, 183758, 175567, 177617, 177619,
#         179667, 175574, 183768, 183780, 183783, 175601, 177653, 177658, 183804, 175613, 175621, 183813, 175623, 183814,
#         177673, 183815, 183821, 183825, 175635, 175636, 183841, 175652, 175662, 183857, 177716, 177717, 177739, 177744,
#         175704, 177759, 181857, 177765, 175718, 175719, 175722, 175724, 175727, 175730, 177784, 175738, 175743, 175745,
#         175753, 175767, 177822, 175819, 179918, 179923, 177881, 177884, 175844, 177892, 175853, 175854, 175855, 177902,
#         177904, 177907, 175860, 177915, 177926, 175879, 175880, 177932, 175886, 177937, 175892, 177947, 175901, 177952,
#         177954, 175915, 177966, 175920, 177972, 177974, 177978, 177979, 180041, 177995, 175963, 175968, 178024, 175978,
#         175981, 178032, 178036, 178042, 176002, 176007, 176015, 178077, 178078, 178079, 176032, 176048, 178103, 178104,
#         176061, 178122, 176088, 176091, 178139, 176096, 178144, 176107, 178156, 178168}
# test_df1 = df[df.partition == 'test']
# cols = test_df1.columns
# dataset = 'bigvul_mix'
# cache_path = cache_dir() / 'data' / dataset / f'{dataset}_cleaned.pkl'
# df = pd.read_pickle(cache_path)
# train_df = df[df.partition == 'train']
# valid_df = df[df.partition == 'valid']
# test_df2 = df[(df.partition == 'test') & (df.vul == 1)][cols]
# print(test_df1.columns, test_df2.columns)
# print(pd.concat([test_df1, test_df2]).duplicated(keep=False))
# cols = ['_id', 'vul', 'func_before']
# a = test_df1[test_df1._id.isin(iiii)][cols]
# b = test_df2[test_df1._id.isin(iiii)][cols]
# print(a)
# print(b)
#
# ret = []
# for i in iiii:
#     aa = a[a._id == i].values[0]
#     bb = b[b._id == i].values[0]
#     print(aa == bb)
#     ret.append(all(aa == bb))
#
# print(all(ret))


dataset = 'bigvul'
cache_path = cache_dir() / 'data' / dataset / f'{dataset}_cleaned.pkl'
df = pd.read_pickle(cache_path)
train_df = df[df.partition == 'train']
valid_df = df[df.partition == 'valid']
test_df1 = df[df.partition == 'test']
test_ds1 = BigVulDatasetDevign(df=test_df1, partition="test", dataset=dataset, vulonly=True)

cols = test_df1.columns
dataset = 'bigvul_mix'
cache_path = cache_dir() / 'data' / dataset / f'{dataset}_cleaned.pkl'
df = pd.read_pickle(cache_path)
train_df = df[df.partition == 'train']
valid_df = df[df.partition == 'valid']
test_df2 = df[(df.partition == 'test')]
test_ds2 = BigVulDatasetDevign(df=test_df2, partition="test", dataset=dataset, vulonly=True)

a = test_ds1[0]

b = test_ds2[0]
print(a.ndata['_WORD2VEC'])
print(b.ndata['_WORD2VEC'])
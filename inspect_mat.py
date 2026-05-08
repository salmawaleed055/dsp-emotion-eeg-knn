import scipy.io as sio

m = sio.loadmat(r"d:/DSP/Project DSP- Spring 2026/Data/s01.mat")
print(m.keys())
for k in ["data", "fs", "labels", "channel_names"]:
    v = m.get(k)
    print(k, type(v), getattr(v, "shape", None))

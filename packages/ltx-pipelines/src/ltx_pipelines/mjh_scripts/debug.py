# import pandas
# import pdb
# from tqdm import tqdm
# csv_file = "/root/autodl-tmp/mjh_proj/MoChaBench-main/benchmark/benchmark.xlsx"
# if csv_file.lower().endswith((".xlsx", ".xls")):
#     ds = pandas.read_excel(csv_file)
# else:
#     ds = pandas.read_csv(csv_file)
# process_bar = tqdm(total=len(ds), desc="Processing rows")
# for _ , row in ds.iterrows():
#     print(row.keys())
#     print(row['context_id'])     
#     process_bar.update(1)


# from PIL import Image
# img_path = "/root/autodl-tmp/mjh_proj/MoChaBench-main/benchmark/first-frames-from-mocha-generation/1p_closeup_facingcamera/1_man_guitar.png"
# img = Image.open(img_path)
# print(img.size)
    
# tmp= (_i for _i, _s in enumerate([1,2,3,4,5,6,8,7,2,7,2]) if _s == 2)
# print(type(tmp))
# print(next(tmp, None))
# print("successful!")

# tmp = (1,2,3,4,5,6,8,7,2,7,2)
# print(next(tmp, None))


import numpy as np
a = [1,2,1,2,1,2,1,2,1,2]
a = np.array(a)
tmp = a.reshape(-1, 2)
# tmp = a.reshape(2, -1)
print(tmp)
print(tmp.shape)
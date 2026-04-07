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


from PIL import Image
img_path = "/root/autodl-tmp/mjh_proj/MoChaBench-main/benchmark/first-frames-from-mocha-generation/1p_closeup_facingcamera/1_man_guitar.png"
img = Image.open(img_path)
print(img.size)
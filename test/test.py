import pandas as pd
import sys
sys.path.append("/home/nick/Preprocessing/src")
from preprocessing import preprocess


data = pd.read_csv("/home/nick/Preprocessing/test/LungCap.csv")
# data["LungCap"] = pd.cut(data["LungCap"], bins=3)

# plant missing values in the data
data["Age"] = data["Age"].sample(frac=0.9, random_state=42)
data["Height"] = data["Height"].sample(frac=0.9, random_state=0)

df = preprocess(
    df=data, 
    outputs=["LungCap"], 
    datetime=None, 
    classification=False,
)

df.to_csv("/home/nick/Regression/test/LungCap.csv", index=False)
# df.to_csv("/home/nick/Classification/test/LungCap.csv", index=False)

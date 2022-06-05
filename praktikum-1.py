print("1. kode akan dimodifikasi menjadi seperti dibawah"):
# import libraries
import cupy as cp
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from scipy import stats
plt.style.use("ggplot")
import warnings
warnings.filterwarnings("ignore")
from scipy import stats

print("2. ya dibuktikan dengan tabel berikut"):
# read data as pandas data frame
url_data = "https://raw.githubusercontent.com/supasonicx/ATA-praktikum-01/main/data.csv"
data = pd.read_csv(url_data)
data = data.drop(['Unnamed: 32','id'],axis = 1)

print("3. tabel diagram korelasi"):
plt.figure(figsize = (15,10))
sns.jointplot(data.compactness_mean,data.concavity_mean,kind="reg")
plt.show()

print("4. nilai covariance dari fitur compactness_mean dengan concavity_mean"):
np.cov(data.compactness_mean,data.concavity_mean)
print("Covariance diantara compactness_mean dan concavity_mean: ",data.compactness_mean.cov(data.concavity_mean))

print("5. Hitung nilai pearson correlation dari fitur compactness_mean dengan concavity_mean"):
p1 = data.loc[:,["compactness_mean","concavity_mean"]].corr(method= "pearson")
p2 = data.concavity_mean.cov(data.compactness_mean)/(data.concavity_mean.std()*data.compactness_mean.std())
print('Pearson correlation: ')
print(p1)
print('Pearson correlation: ',p2)

print("6. Lakukan uji hipotesis untuk kolom fitur compactness_mean dengan concavity_mean yang berbeda serta berikan penjelasan terhadap hasil dari uji hipotesis yang dilakukan."):
statistic, p_value = stats.ttest_rel(data.compactness_mean,data.concavity_mean)
print('p-value adalah: ',p_value)

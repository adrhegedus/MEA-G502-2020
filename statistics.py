# %%
from pylab import rcParams
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd


folder = 'linear'
foderOut = 'out'
directory = os.getcwd() + '//' + folder
path = directory + '\\' + foderOut

#

data = []


for index, filename in enumerate(os.listdir(directory)):

    if(filename.endswith(".csv")):
        print(index)
        try:

            curr = pd.read_csv(directory + '//' + filename)

        except:
            print("-----------------could not read data----------------")

        if curr.empty != True:
            length = len(curr)
            index = list([index]*length)
            print("Dataframe is not empty")

        else:
            print("Dataframe is empty")

        curr['ID'] = index
        data.append(curr)

data = pd.concat(data)
data.fillna(0, inplace=True)

data.to_csv("linear/out/AllData.csv", index=False, encoding='utf-8-sig')


print("-----------------DATA------------------", '\n', data)

# %%

# to get average value of total time played
totalTime = data.copy()

totalTime = totalTime["total time played"]

type(totalTime)

df = totalTime.to_frame()

df = df[(df['total time played'] != '-')]

df["total time played"] = pd.to_numeric(
    df["total time played"], downcast="float")

df.mean()
df.to_csv("df_linear_totalTimePlayed.csv", index=False, encoding='utf-8-sig')



# %%
import seaborn as sns

bins_list = [ 20, 40, 50, 60, 80, 110]
plt.boxplot(df)
plt.show()

# %% 
sns_plot = sns.displot(df, x="total time played",bins=10, kde=True)
# sns_plot.savefig('adaptive/out/adaptive_total_time_played.png')

sns_plot.savefig('adaptive/out/adaptive_total_time_played.png')
# %%
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

# %% 

x = data['x-position']
y = data['z-position']

heatmap, xedges, yedges = np.histogram2d(x, y, bins=150)

X = np.divide(heatmap, heatmap.max())
X = np.ma.masked_where(X == 0, X)

rcParams['figure.figsize'] = 20, 16

extent = [-1300, 1300, -1300, 1300]

map_img = mpimg.imread('linear/background.png')
plt.clf()

# hmax.collections[0].set_alpha(0)
plt.imshow(map_img, zorder=0, extent=[-1300, 1300, -1300, 1300], alpha=1)
img = plt.imshow(X.T, extent=extent, origin='lower',
           alpha=1, vmin=0, vmax=1, cmap="Reds")

cmap = colors.ListedColormap(['white', 'red'])
bounds=[0,0.2,0.5,0.7 ,1]
norm = colors.BoundaryNorm(bounds, cmap.N)

plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=[0,0.2,0.5,0.7 ,1])
# plt.show()
plt.savefig('foo_adaptive.png')

# %%

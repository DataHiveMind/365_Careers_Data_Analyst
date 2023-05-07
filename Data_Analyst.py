# -------------------------------------------------------------------------------------- Basic Structures
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
sns.set()

if 5 == 15/3:
    print("Hooray!")

x = 1
if x > 3:
    print("Case 1")
else:
    print("Case 2")


def compare_to_five(y):
    if y > 5:
        return "Greater"
    elif y < 0:
        return "Negative"
    elif y < 5:
        return "Less"
    else:
        return "Equal"


people = ["ken", "Luis", "brandon"]
print(people[2])

Numbers = [1, 2, 3, 4, 5]
Numbers.sort()
Numbers

Numbers.sort(reverse=True)
Numbers

# Tuple
(age, years) = "30, 17". split(',')
print("Age: " + age)
print("years: " + years)


def square_info(x):
    A = x ** 2
    p = 4 * x
    print("Area and Perimeter")
    return A, p


square_info(3)

dict = {'k1': "cat", 'k2': "dog", 'k3': "mouse", 'k4': 'fish'}
dict

# For Loops
even = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
for n in even:
    print(n, end=" ")

for x in range(20):
    if x % 2 == 0:
        print(x, end=" ")
    else:
        print("odd", end=" ")

# ------------------------------------------------------------------------------------Mathmatics

# Vector
v = np.array([5, -2, 4])
v

# Matrices
n = np.array([[5, 12, 6], [-3, 0, 14]])
n

# Tensor
m1 = np.array([[5, 12, 6], [-3, 0, 14]])
m2 = np.array([[9, 8, 7], [1, 3, -5]])
t = np.array([m1, m2])
print(t)

# ------------------------------------------------------------------------------tranpose
# transpose vector
x = np.array([1, 2, 3])
x.T

# transpose scalars
s = np.array([3])
s.T

# ----------------------------------------------------------------------------Dot Product
# Vector times vector
x = np.array([2, 8, -4])
y = np.array([1, -7, 3])

np.dot(x, y)

z = np.array([0, 2, 5, 8])
k = np.array([20, 3, 4, -1])

np.dot(z, k)

# -------------------------------------------------------------------------- NumPy basics

array_a = np.array([1, 2, 3])
array_b = np.array([[1, 2, 3], [4, 5, 6]])

print(array_a, array_b)

# -----------------------------------------------------------------------------Pandas basics

products = ['A', 'B', 'C', 'D']
prodtype = pd.Series(products)
prodtype

array = np.array([10, 20, 30, 40, 50])

SeriesA = pd.Series(array)
SeriesA

# --------------------------------------------------------------------------------Text Data
product = "AB"
print('item: "%s".' % product)

prod = ['A', 'B']
print('item: '"%s" % prod)
print('item: '"%s" % prod[1])

print('This product is from cat "A".\rProduct 01 ')

s = "Price per unit"
s1 = s.replace("Price", "Cost")
s1

s1.startswith('Cost')
s1.split()
s1.split(' ', maxsplit=0)

time_horizon = 1, 3, 12
products = ['product A', 'Product B']

'Expected slaes for a period of {} month(s) for {}:'.format(
    time_horizon[2], products[1])

t = (4, 5, 6, 7)
l = [10, 20, 30, 40]
s = "abcd"

for i in t:
    print(i, end=" ")

range(0, 5)

for i in range(5):
    print(i, end=" ")

numbers = [1, 13, 4, 5, 63, 100]
new_numbers = []
for n in numbers:
    new_numbers.append(n * 2)

print(new_numbers)

for i in range(2):
    for j in range(5):
        print(i + j, end=" ")

# ------------------------------------------------------------ pandas Series by .unique() and .nunique()
data = pd.read_csv('Location.csv', squeeze=True)
location_data = data.copy()
location_data.head()

type(location_data)
len(location_data)

location_data.describe()
location_data.nunique()
location_data.unique()

# ----------------------------------------------------------------------- pandas dataframes

array_a = np.array([[3, 2, 1]], [[6, 3, 2]])
pd.DataFrame(array_a)

type(pd.DataFrame(array_a))

df = pd.DataFrame(array_a, columns=['Column 1', 'Column 2', 'Column 3'])
df

df = pd.DataFrame(array_a, columns=[
                  'Column 1', 'Column 2', 'Column 3'], index=['Row 1', 'Row 2'])
df

data = pd.read_csv('lending-company.csv', index_col='LoanID')
lending_co_data = data.copy()
lending_co_data.head()

type(lending_co_data)

lending_co_data['Product']
lending_co_data.head()

# -------------------------------------------------------------------------Numpy Fundementals

array = np.array([[1, 2, 3], [4, 5, 6]])
array[0]
array[0][1]

array[0, 2] = 9
array[0] = 4
array[:, 0] = 6

list = {0, 7, 8}
array[0] = list

# -----------------------------------------------------------------------------------Data Visualization
sns.set()

# -------------------------------------------------bar chart
dfcar = pd.read_csv("Bar_chart_data.csv")

plt.figure(figsize=(9, 6))
plt.bar(x=dfcar["Brand"],
        height=dfcar["Cars Listings"],
        color="rgbwymc")
plt.title("Cars ListingS by Brand", fontsize=16, fontweight="bond")
plt.ylabel("Number of Listings", fontsize=13)
plt.xticks(rotation=45)
plt.show()

plt.savefig("Used Cars Bar.png")

# -----------------------------------------------------pie chart
dffuel = pd.read_csv("pie_chart_data.csv")

sns.setpalette('colorblind')

plt.pie(dffuel["Number of cars"],
        labels=dffuel["Engine type"].values,
        autopct="%.2f%%", textprops={'size': 'x-large', 'fontweight': 'bold', 'rotation': '30', 'color': 'w'})
plt.legend()
plt.title("Cars by Engine fuel", fontsize=16, fontweight='bold')
plt.show()
# -----------------------------------------------------stacked chat
fuel = pd.read_csv("stacked_area_data.csv")
labels = ["Diesel", "Patrol", "gas"]
plt.stackplot(fuel["Year"], fuel["Diesel"], fuel["Patrol"])
plt.xticks(fuel["year"], rotation=45)
plt.legend(labels=labels, loc="upper left")
plt.ylabel("Number of cars", fontsize=13)
plt.show()

# ---------------------------------------------------------line chart
dfl = pd.read_csv("line_chart_data.csv")
plt.plot(dfl["new-data"], dfl["GSPC500"])
plt.plot(dfl['new_data'], dfl["FTSE100"])
plt.ylabel("Returns")
plt.xlabel("Dtae")
plt.show()

# -----------------------------------------------------------histogram Chart
df = pd.read_csv("histogram data.csv")
sns.set_style("White")
plt.figure(figsize=(8, 6))
plt.hist(df["price"], bins=8)
plt.show()

# ----------------------------------------------------------------Scatter plot
df = pd.read_csv("scatterplot.csv")
plt.scatter(df["Area(ft.)"], df['price'], alpha=0.6, cmap="virdis")
sns.scatterplot(df["price"], df["Area (ft.)"],
                hue=df["type"], palette=["black", "pink"])
plt.show()

# -------------------------------------------------------------------Regression plot
df = pd.read_csv("Regression.csv")
sns.regplot(x="Budget", y="sales", data=df, scatter_kws={"color"
                                                         "k"}, line_kws={"color": "red"})
sns.set(rc={'figure.figszie': (9, 6)})

plt.show()

# ----------------------------------------------------------------------bar and line chart
df = pd.read_csv("barandlinechart.csv")

fig, ax = plt.subplots(figsize=(10, 7))
ax.bar(df["year"], df["particpants"], color="k")
ax1 = ax.twinx()
ax1.plot(df["year"], df["python Users"], marker="D")

ax1.set_ylim(0, 1)
ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
ax.set_ylabel("Num of people", weight="Bold")

ax1.tick_params(axis="y", width=2, labelsize="large")

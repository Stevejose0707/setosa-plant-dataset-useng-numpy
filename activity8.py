import numpy as np
url='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
q=np.genfromtxt(url,delimiter=',',usecols=(0,1,2,3,4))
x=np.nan_to_num(q,nan=0)
print(x)
petal=q[:,2].astype(float)
conditions=[(petal< 3),(petal>= 3) & (petal< 5),(petal>= 5)]
choices=['smaall','medium','large']
petal_size=np.select(conditions,choices,default='unknown')
print(petal_size[:10])
#What is the value of second longest petal length of species setosa
species=q[:,4]
setosa_petallength=petal[species=='Iris-setosa']
sorted_setosa_length=np.sort(setosa_petallength)[::-1]
second=sorted_setosa_length[1]
print(second)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
data = np.genfromtxt(url, delimiter=",", dtype="str", usecols=(0, 1, 2, 3, 4))  # Include species column

# Extract species and petal length
species = data[:, 4]  # Species is in 5th column (index 4)
petal_length = data[:, 2].astype(float)  # Petal length is in 3rd column (index 2), convert to float

# Filter data for Iris-setosa species
setosa_petal_length = petal_length[species == "Iris-setosa"]

# Sort in descending order
v = np.sort(setosa_petal_length)[::-1]

# Get the second longest petal length
second = v[1]

# Print the result
print("The second longest petal length of Iris-setosa is:", second)
import pandas as pd
import matplotlib.pyplot as plt

data = {
    'Survived': [0, 1, 0, 1, 0, 1, 0, 0, 1, 0], 
    'SibSp': [1, 0, 3, 0, 0, 1, 0, 2, 1, 0],
    'Parch': [0, 2, 1, 1, 0, 1, 0, 0, 2, 0],
    'Fare': [7.25, 71.83, 8.05, 53.1, 8.46, 12.29, 9.5, 15.5, 26.55, 7.75]  # Fare price
}

df = pd.DataFrame(data)
df['Number of Relatives'] = df['SibSp'] + df['Parch']
dead_passengers = df[df['Survived'] == 0]

plt.figure(figsize=(8, 5))
plt.scatter(dead_passengers['Number of Relatives'], dead_passengers['Fare'], color='red', alpha=0.6)
plt.xlabel('Number of Relatives')
plt.ylabel('Fare')
plt.title('Scatter Plot of Dead Passengers (Number of Relatives vs Fare)')
plt.grid(True)
plt.show()

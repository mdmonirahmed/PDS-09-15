import pandas as pd
import matplotlib.pyplot as plt

data_url = "https://raw.githubusercontent.com/PulockDas/pd-12-resources/refs/heads/master/titanic.csv"
df = pd.read_csv(data_url)

missing_cols = df.columns[df.isnull().any()].tolist()
print("Columns with null values:", missing_cols)

df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Cabin'].fillna('Unknown', inplace=True)

survival_counts = df.groupby(['Survived', 'Sex']).size().unstack()
survival_counts.plot(kind='bar', stacked=True, color=['blue', 'pink'])
plt.title('Survival Count by Gender')
plt.xlabel('Survived (0 = Dead, 1 = Survived)')
plt.ylabel('Count')
plt.legend(title='Gender')
plt.show()

pclass_counts = df.groupby(['Survived', 'Pclass']).size().unstack()
pclass_counts.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Survival Count by Passenger Class')
plt.xlabel('Survived (0 = Dead, 1 = Survived)')
plt.ylabel('Count')
plt.legend(title='Pclass')
plt.show()

def age_class(age):
    if age <= 16:
        return 0
    elif age <= 26:
        return 1
    elif age <= 36:
        return 2
    elif age <= 62:
        return 3
    else:
        return 4

df['AgeClass'] = df['Age'].apply(age_class)
df.drop(columns=['Age'], inplace=True)
ageclass_counts = df.groupby(['Survived', 'AgeClass']).size().unstack()
ageclass_counts.plot(kind='bar', stacked=True, colormap='coolwarm')
plt.title('Survival Count by Age Class')
plt.xlabel('Survived (0 = Dead, 1 = Survived)')
plt.ylabel('Count')
plt.legend(title='AgeClass')
plt.show()

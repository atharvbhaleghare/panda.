import seaborn as sns
import matplotlib.pyplot as plt

# Load sample dataset (Titanic)
df = sns.load_dataset('titanic')[['age', 'fare', 'pclass', 'survived']].dropna()

# Histogram
sns.histplot(df['age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

# Scatter plot
sns.scatterplot(x='age', y='fare', hue='survived', data=df)
plt.title("Age vs Fare by Survival")
plt.show()

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

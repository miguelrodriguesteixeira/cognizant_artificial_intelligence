import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Specify the path to your CSV file
path = "/Users/miguelteixeira/cognizant/Cognizant - Data Analysis/sample_sales_data.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(path)

# Drop the "Unnamed: 0" column if it exists
df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')

# Display basic information about the DataFrame
df_info = df.info()

# Display summary statistics for numerical columns
summary_stats = df.describe()

# Count of null values in each column
null_counts = df.isnull().sum()

def plot_continuous_distribution(data: pd.DataFrame = None, column: str = None, height: int = 8):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(data[column], kde=True, ax=ax)
    ax.set_title(f'Distribution of {column}')
    plt.show()

def get_unique_values(data, column):
    num_unique_values = len(data[column].unique())
    value_counts = data[column].value_counts()
    print(f"Column: {column} has {num_unique_values} unique values\n")
    print(value_counts)

def plot_categorical_distribution(data: pd.DataFrame = None, column: str = None, height: int = 8, aspect: int = 2):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(data=data, x=column, ax=ax)
    ax.set_title(f'Distribution of {column}')
    plt.show()

def correlation_plot(data: pd.DataFrame = None):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title('Correlation Matrix')
    plt.show()

# Scatter plot of unit price and quantity
def scatter_plot(data: pd.DataFrame = None, x_column: str = None, y_column: str = None):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=data, x=x_column, y=y_column, ax=ax)
    ax.set_title(f'Scatter Plot: {x_column} vs {y_column}')
    plt.show()

# Bar plot of total sales per category
def bar_plot(data: pd.DataFrame = None, x_column: str = None, y_column: str = None):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=data, x=x_column, y=y_column, ax=ax)
    ax.set_title(f'Bar Plot: {y_column} per {x_column}')
    plt.show()

# Pair plot for numerical columns
def pair_plot(data: pd.DataFrame = None):
    sns.pairplot(data)
    plt.suptitle('Pair Plot of Numerical Columns', y=1.02)
    plt.show()

# Print summary and insights
print("\nExploratory Data Analysis Summary:")
print("1. Basic Information:")
print(df_info)
print("\n2. Summary Statistics:")
print(summary_stats)
print("\n3. Null Value Counts:")
print(null_counts)

# Plot distributions and correlations
plot_continuous_distribution(data=df, column='unit_price')
plot_categorical_distribution(data=df, column='category')
correlation_plot(data=df)

# Scatter plot of unit price and quantity
scatter_plot(data=df, x_column='unit_price', y_column='quantity')

# Bar plot of total sales per category
bar_plot(data=df, x_column='category', y_column='total')

# Pair plot for numerical columns
pair_plot(data=df)

# Provide recommendations for next steps
print("\nRecommendations for Next Steps:")
print("1. Increase Data Volume:")
print("   - Gather data from multiple stores and over a more extended period to generalize insights.")
print("2. Refine Problem Statement:")
print("   - Define a more specific problem statement related to inventory management or sales optimization.")
print("3. Feature Expansion:")
print("   - Identify additional features that could help in better understanding customer behavior or product trends.")

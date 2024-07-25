import pandas as pd

numbers = list(range(1, 11))

# Creating a DataFrame with a column named 'nums'
df = pd.DataFrame(numbers, columns=['nums'])

def convert(x):
    if x > 5:
        return "large"
    else:
        return "small"

df["new"] = df["nums"].apply(convert)
print(df)
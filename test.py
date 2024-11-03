from sklearn import datasets 


digits_df = datasets.load_digits()

print(len(digits_df['target']))
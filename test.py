import pandas as pd

dic = {'Bruins' : 1, 'Boston' : 34}

df = pd.DataFrame(dic.items(), columns = ['Team', 'Elo']).sort_values('Elo', ascending=False)

print(df)

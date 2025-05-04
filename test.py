import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_iqr(column: pd.Series) -> tuple[float, float]:
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1

    lower_q = Q1 - 1.5 * IQR
    upper_q = Q3 + 1.5 * IQR
    
    return lower_q, upper_q

def get_irq_stat(column: pd.Series) -> float:
    l, u = calculate_iqr(column)
    
    cnt = len(column[(column < l) | (column > u)])
    
    return cnt / len(column)


def main() -> None:
    origin_df = pd.read_csv('data/AB_NYC_2019.csv')

    print(origin_df.columns)

    column = origin_df["price"]
    l, u = calculate_iqr(column)
    print(get_irq_stat(column) * 100, column.min(), column.max())
    
    print(origin_df[column > 1000])
    
    _, ax = plt.subplots(nrows=3, ncols=1)

    ax[0].hist(column, bins=100, alpha=0.5, edgecolor='black')
    ax[0].axvline(x=l, color='green', linestyle='--')
    ax[0].axvline(x=u, color='green', linestyle='--')
    ax[0].axvline(x=column.mean(), color='red', linestyle='--')

    col = column[(column >= l) & (column <= u)]
    ax[1].hist(col, bins=100, alpha=0.5, edgecolor='black')
    ax[1].axvline(x=col.mean(), color='red', linestyle='--')

    col = column[column > u]
    ax[2].hist(col, bins=100, alpha=0.5, edgecolor='black')

    plt.show()


if __name__ == "__main__":
    main()

from plot import get_all_datasets
import pandas as pd

def select_by_step(data, step):
    frames = []
    for i in range(len(data)):
        # print(data[0])
        # print(data[i][data[i]['TotalEnvInteracts'] == 220000])
        frames.append(data[i][data[i]['TotalEnvInteracts'] == step])
    result = pd.concat(frames, ignore_index=True)
    return result

def cal_mean(data, col_name = 'AverageTestEpRet'):
    if not col_name in data.columns:
        col_name = 'AverageEpRet' if col_name == 'AverageTestEpRet' else 'AverageTestEpRet'
    # print(data[[col_name,'Condition1']])
    result_groupby = data[[col_name,'Condition1']].groupby(by=["Condition1"])
    result = pd.concat([round(result_groupby.mean(),2) ,round(result_groupby.std(),2)], axis=1, ignore_index=True)
    print(result)
    return result

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--step', default=1000000)
    parser.add_argument('--col_name', default='AverageTestEpRet')
    args = parser.parse_args()
    """

    Args: 
        logdir (strings): As many log directories (or prefixes to log 
            directories, which the plotter will autocomplete internally) as 
            you'd like to plot from.

        step (number): Pick the values of this step evaluation data.

        col_name (strings): Pick what columns from data to calculate 
            its mean and standard divation.
    """


    data = get_all_datasets(args.logdir)
    selected = select_by_step(data, int(args.step))
    cal_mean(selected, args.col_name)

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from scipy import stats as sps
from scipy.interpolate import interp1d

GAMMA = 1/7
R_T_MAX = 12
r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)

def get_posteriors(sr, sigma=0.15):

    # (1) Calculate Lambda
    lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))
    lam += 10e-6

    # (2) Calculate each day's likelihood
    likelihoods = pd.DataFrame(
        data = sps.poisson.pmf(sr[1:].values, lam),
        index = r_t_range,
        columns = sr.index[1:])

    # (3) Create the Gaussian Matrix
    process_matrix = sps.norm(loc=r_t_range,
                              scale=sigma
                             ).pdf(r_t_range[:, None])

    # (3a) Normalize all rows to sum to 1
    process_matrix /= process_matrix.sum(axis=0)

    # (4) Calculate the initial prior
    prior0 = sps.gamma(a=4).pdf(r_t_range)
    prior0 /= prior0.sum()

    # Create a DataFrame that will hold our posteriors for each day
    # Insert our prior as the first posterior.
    posteriors = pd.DataFrame(
        index=r_t_range,
        columns=sr.index,
        data={sr.index[0]: prior0}
    )

    # We said we'd keep track of the sum of the log of the probability
    # of the data for maximum likelihood calculation.
    log_likelihood = 0.0

    # (5) Iteratively apply Bayes' rule
    for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):

        #(5a) Calculate the new prior
        current_prior = process_matrix @ posteriors[previous_day]

        #(5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
        numerator = likelihoods[current_day] * current_prior

        #(5c) Calcluate the denominator of Bayes' Rule P(k)
        denominator = np.sum(numerator)

        # Execute full Bayes' Rule
        posteriors[current_day] = numerator/denominator

        # Add to the running sum of log likelihoods
        log_likelihood += np.log(denominator)

    return posteriors, log_likelihood


def prepare_cases(cases, state_name, latest, cutoff=0):

    new_cases = cases.diff()

    # fill until latest
    state = new_cases.xs(state_name)
    if latest not in state:
        new_cases = new_cases.append(pd.Series({(state_name, latest): 0}))

    # fill NaN
    first = new_cases.index[0]
    new_cases.loc[first] = cases[0]

    # fill 0
    new_cases = new_cases.unstack(level=[0]).asfreq('D', fill_value=0).stack(level=[0]).swaplevel(1,0)

    std = 2
    window = 7
    if new_cases.values.max() < 5:
        std = 0.1
    elif new_cases.values.max() < 25:
        window = 5

    smoothed = new_cases.rolling(window,
        win_type='gaussian',
        min_periods=1,
        center=True).mean(std=std).round()

    idx_start = np.searchsorted(smoothed, cutoff)

    smoothed = smoothed.iloc[idx_start:]
    original = new_cases.loc[smoothed.index]

    return original, smoothed


def highest_density_interval(pmf, p=.9):
    # If we pass a DataFrame, just call this recursively on the columns
    if(isinstance(pmf, pd.DataFrame)):
        return pd.DataFrame([highest_density_interval(pmf[col], p=p) for col in pmf],
                            index=pmf.columns)

    cumsum = np.cumsum(pmf.values)

    # N x N matrix of total probability mass for each low, high
    total_p = cumsum - cumsum[:, None]

    # Return all indices with total_p > p
    lows, highs = (total_p > p).nonzero()

    # Find the smallest range (highest density)
    best = (highs - lows).argmin()

    low = pmf.index[lows[best]]
    high = pmf.index[highs[best]]

    return pd.Series([low, high],
                     index=[f'Low_{p*100:.0f}',
                            f'High_{p*100:.0f}'])


url = 'data/tokyo_patient_diff.csv'
data = pd.read_csv(url,
                     usecols=['発表日', '区市町村', '人数'],
                     parse_dates=['発表日'],
                    ).sort_index()

FILTERED_REGION = []

PREF =  ['千代田区', '中央区', '港区', '新宿区', '文京区', '台東区', '墨田区', '江東区', '品川区', '目黒区', '大田区', '世田谷区', '渋谷区', '中野区', '杉並区', '豊島区', '北区', '荒川区', '板橋区', '練馬区', '足立区', '葛飾区', '江戸川区', '八王子市', '立川市', '武蔵野市', '三鷹市', '青梅市', '府中市', '昭島市', '調布市', '町田市', '小金井市', '小平市', '日野市', '東村山市', '国分寺市', '国立市', '福生市', '狛江市', '東大和市', '清瀬市', '東久留米市', '武蔵村山市', '多摩市', '稲城市', '羽村市', 'あきる野市', '西東京市', '瑞穂町', '日の出町', '檜原村', '奥多摩町', '大島町', '利島村', '新島村', '神津島村', '三宅村', '御蔵島村', '八丈町', '青ヶ島村', '小笠原村']

df_all = pd.DataFrame()
for pref in PREF:
    ds = data.groupby(['発表日', '区市町村'], as_index=False).sum()
    dsp = ds[ds['区市町村']==pref]
    dsp['累積人数'] = dsp['人数'].cumsum()
    dsp = dsp.rename(columns={'発表日': 'date', '区市町村': 'pref'})
    dspi = dsp.set_index(['pref', 'date'])
    df_all = pd.concat([df_all, dspi])
states = df_all['累積人数']



# Choosing the optimal sigma
sigmas = np.linspace(1/20, 1, 20)

targets = ~states.index.get_level_values('pref').isin(FILTERED_REGION)
states_to_process = states.loc[targets]

results = {}

for state_name, cases in states_to_process.groupby(level='pref'):

    tokyo = states.xs('新宿区')
    latest = tokyo.index[-1]

    print(state_name)
    new, smoothed = prepare_cases(cases, state_name, latest, cutoff=0)

    result = {}

    # Holds all posteriors with every given value of sigma
    result['posteriors'] = []

    # Holds the log likelihood across all k for each value of sigma
    result['log_likelihoods'] = []

    for sigma in sigmas:
        posteriors, log_likelihood = get_posteriors(smoothed, sigma=sigma)
        result['posteriors'].append(posteriors)
        result['log_likelihoods'].append(log_likelihood)

    # Store all results keyed off of state name
    results[state_name] = result

print('Done.')

total_log_likelihoods = np.zeros_like(sigmas)

# Loop through each state's results and add the log likelihoods to the running total.
for state_name, result in results.items():
    total_log_likelihoods += result['log_likelihoods']

# Select the index with the largest log likelihood total
max_likelihood_index = total_log_likelihoods.argmax()

# Select the value that has the highest log likelihood
sigma = sigmas[max_likelihood_index]
print(f'Maximum Likelihood value for sigma = {sigma:.2f}')


# Compile Final Results
final_results = None

for state_name, result in results.items():
    print(state_name)
    posteriors = result['posteriors'][max_likelihood_index]
    hdis_90 = highest_density_interval(posteriors, p=.9)
    hdis_50 = highest_density_interval(posteriors, p=.5)
    most_likely = posteriors.idxmax().rename('ML')
    result = pd.concat([most_likely, hdis_90, hdis_50], axis=1)
    if final_results is None:
        final_results = result
    else:
        final_results = pd.concat([final_results, result])

print('Done.')

# Export
final_results.to_csv('dist/rt_tokyo.csv', float_format='%.2f')

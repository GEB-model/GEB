import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import numpy as np

from pgmpy.estimators import K2Score
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import State

from pgmpy.estimators import PC, HillClimbSearch, ExhaustiveSearch

from scipy.stats import chi2_contingency

from preconfig import ORIGINAL_DATA, PREPROCESSING_FOLDER

def load_farmer_survey(password):
    # Read the survey data
    with zipfile.ZipFile(ORIGINAL_DATA / 'survey_results_cleaned.zip') as zf:
        with zf.open('survey_results_cleaned.xlsx', pwd=password) as excel_file:
            df = pd.read_excel(excel_file)
    return df

def load_IHDS_survey():
    df = pd.read_csv(PREPROCESSING_FOLDER / 'agents' / 'farmers' / 'IHDS_I.csv')
    return df

bins = {
    'What is your age?': {
        'bins': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'labels': ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '>90'],
    },
    'Perceived self efficacy': {
        'bins': [1, 1.5, 2.5, 3.5, 4.5, 5],
        'labels': [1, 2, 3, 4, 5],
    },
    'Perceived effectivity': {
        'bins': [1, 1.5, 2.5, 3.5, 4.5, 5],
        'labels': [1, 2, 3, 4, 5],
    },
    "How large is the area you grow crops on in hectares?" : {
        'bins': [0, 1, 2, 4, 10, np.inf],
        'labels': ['0-1', '1-2', '2-4', '4-10', '>10'],
    },
}

def parse_survey(df):
    df['Perceived self efficacy'] = df[
        [column for column in df.columns if column.startswith('Ability - ')]
    ].mean(axis=1)

    df['Perceived effectivity'] = df[
            [column for column in df.columns if column.startswith('Effectivity - ')]
    ].mean(axis=1)
    df["How large is the area you grow crops on in hectares?"] = df["How large is the area you grow crops on in acres?"] * 0.404686
    
    for question, values in bins.items():
        assert len(values['bins']) == len(values['labels']) + 1, "Bin bounds must be one longer than labels"
        df[question] = pd.cut(df[question], bins=values['bins'], labels=values['labels'])

    samples = df[[
        # 'What is your gender?',
        'How many years of education did you complete?',
        # 'Savings',
        # 'Loans',
        'What is your age?',
        'Which sources do you use for irrigation?',
        "How large is the area you grow crops on in hectares?",
        "Are you planning to adopt any additional drought adaptation measures in the coming five years?",
        # "In which section of the survey area does the surveyee live?",
        # "Which crops did you grow during the last Kharif season?",
        "Perceived self efficacy",
        "Perceived effectivity"
    ]]
    # remove where data is -1
    samples = samples[(samples != -1).all(1)]
    # remove where data is NaN
    samples = samples.dropna()

    # replace all spaces in column names with underscores, otherwise pgmpy will throw an error when saving/loading model
    samples.columns = samples.columns.str.replace(' ', '_')
    # remove all ?
    samples.columns = samples.columns.str.replace('?', '')
    # assert all column names are valid with values -0-9A-Z_a-z
    assert all(samples.columns.str.match(r'^[0-9A-Za-z_]+$'))

    # replace all spaces in dataframe data with underscores, otherwise pgmpy will throw an error when saving/loading model
    samples = samples.replace(' ', '_', regex=True)
    return samples

def structure_learning(samples, max_indegree=3):
    print("Estimating network structure")
    est = HillClimbSearch(data=samples)
    model = est.estimate(
        scoring_method=K2Score(data=samples),
        max_indegree=max_indegree,
        max_iter=int(1e4),
        epsilon=1e-8,
    )
    return model

def estimate_parameters(model, samples, plot=False):
    print("Learning network parameters")
    model = BayesianNetwork(model.edges())
    model.fit(
        samples, estimator=BayesianEstimator, prior_type="dirichlet", pseudo_counts=0.1
    )
    model.get_cpds()

    edge_params = {}
    for edge in model.edges():
        # get correlation between two variables
        cross_tab = pd.crosstab(samples[edge[0]], samples[edge[1]])
        chi_stat = chi2_contingency(cross_tab)[0]
        N = len(samples)
        minimum_dimension = (min(cross_tab.shape)-1)

        # Cramer’s V value
        cramers_v_value = np.sqrt((chi_stat/N) / minimum_dimension)

        edge_params[edge] = {'label': round(cramers_v_value, 2)}

    if plot:
        model.to_daft('circular', edge_params=edge_params, pgm_params={'grid_unit': 10}, latex=False).render()
        plt.show()
    return model

def sample(model, n, evidence=[]):
    print("Generating samples")
    sampler = BayesianModelSampling(model)
    # if no evidence this is equalivalent to forward sampling
    sample = sampler.rejection_sample(evidence=evidence, size=n, show_progress=True)
    return sample

def save(model, path):
    print("Saving model")
    model.save(str(path))

def load(path):
    print("Loading model")
    return BayesianNetwork.load(str(path), filetype='bif')


if __name__ == '__main__':
    bayesian_net_folder = PREPROCESSING_FOLDER / 'bayesian_net'
    bayesian_net_folder.mkdir(exist_ok=True)
    bayesian_net_path = bayesian_net_folder / 'bayesian_net.bif'

    # IHDS_survey = load_IHDS_survey()
    farmer_survey = load_farmer_survey(b'2!hM0t$2Kd66')    
    
    samples = parse_survey(farmer_survey)
    model = structure_learning(samples, max_indegree=2)
    model = estimate_parameters(model, samples, plot=False)
    
    save(model, bayesian_net_path)

    del model

    # evidence = [
    #     State(var="How_large_is_the_area_you_grow_crops_on_in_hectares", state='0-1'),
    # ]
    # model = load(bayesian_net_path)
    # sample = sample(model, 1000, evidence=evidence)

    # # sample.to_excel(PREPROCESSING_FOLDER / 'sample.xlsx')

    # print(sample)
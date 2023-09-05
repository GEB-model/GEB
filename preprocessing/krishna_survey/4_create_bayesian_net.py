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


class Survey:
    def learn_structure(self, max_indegree=3):
        print("Estimating network structure")
        est = HillClimbSearch(data=self.samples)
        self.model = est.estimate(
            scoring_method=K2Score(data=self.samples),
            max_indegree=max_indegree,
            max_iter=int(1e4),
            epsilon=1e-8,
        )

    def estimate_parameters(self, plot=False, save=False):
        print("Learning network parameters")
        self.model = BayesianNetwork(self.model.edges())
        self.model.fit(
            self.samples, estimator=BayesianEstimator, prior_type="dirichlet", pseudo_counts=0.1
        )
        self.model.get_cpds()

        edge_params = {}
        for edge in self.model.edges():
            # get correlation between two variables
            cross_tab = pd.crosstab(self.samples[edge[0]], self.samples[edge[1]])
            chi_stat = chi2_contingency(cross_tab)[0]
            N = len(self.samples)
            minimum_dimension = (min(cross_tab.shape)-1)

            # Cramer’s V value
            cramers_v_value = np.sqrt((chi_stat/N) / minimum_dimension)

            edge_params[edge] = {'label': round(cramers_v_value, 2)}

        if plot or save:
            self.model.to_daft('circular', edge_params=edge_params, pgm_params={'grid_unit': 10}, latex=False).render()
            if save:
                plt.savefig(save)
            if plot:
                plt.show()

    def fix_naming(self):
        # replace all spaces in column names with underscores, otherwise pgmpy will throw an error when saving/loading model
        self.samples.columns = self.samples.columns.str.replace(' ', '_')
        # remove all ?
        self.samples.columns = self.samples.columns.str.replace('?', '')
        # assert all column names are valid with values -0-9A-Z_a-z
        assert all(self.samples.columns.str.match(r'^[0-9A-Za-z_]+$'))
        # replace all spaces in dataframe data with underscores, otherwise pgmpy will throw an error when saving/loading model
        self.samples = self.samples.replace(' ', '_', regex=True)

    def save(self, path):
        print("Saving model")
        self.model.save(str(path))

    def sample(self, n, evidence=[], evidence_columns=None, method='rejection'):
        assert method == 'rejection', "Only rejection sampling is implemented"
        print("Generating samples")
        sampler = BayesianModelSampling(self.model)
        # if no evidence this is equalivalent to forward sampling
        if evidence:
            assert evidence_columns, "If evidence is given, evidence_columns must be given as well"
            assert len(evidence) == len(evidence_columns), "Number of evidence values must match number of evidence columns"
            for state, evidence_column in zip(evidence, evidence_columns):
                assert state in self.model.states[evidence_column], f"State {state} is not a valid state for variable {evidence_column}"
            evidence = [
                State(var=evidence_column, state=state)
                for evidence_column, state in zip(evidence_columns, evidence)
            ]
        sample = sampler.rejection_sample(evidence=evidence, size=n, show_progress=True)
        return sample

    @property
    def variables(self):
        return self.samples.columns.tolist()

class FarmerSurvey(Survey):
    def __init__(self, password):
        self.password = password
        self.bins = {
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
                'bins': [0, 1, 2, 4, np.inf],
                'labels': ['0-1', '1-2', '2-4', '>4'],  # after size class groups in agricultural census
            },
            'How many years of education did you complete?' : {
                'bins': [-np.inf, 0, 5, 10, 12, np.inf],
                'labels': ['none', '5th_standard', 'matriculation', 'higher_secondary', 'graduate']
            }                
        }
        self.renames = {
            'What is your age?': 'age',
            'How many years of education did you complete?': 'education',
            'How large is the area you grow crops on in hectares?': 'field_size',
        }
        self.survey = self.load()

    def load(self):
        # Read the survey data
        with zipfile.ZipFile(ORIGINAL_DATA / 'survey_results_cleaned.zip') as zf:
            with zf.open('survey_results_cleaned.xlsx', pwd=self.password) as excel_file:
                df = pd.read_excel(excel_file)
        return df

    def parse(self):
        self.survey['Perceived self efficacy'] = self.survey[
            [column for column in self.survey.columns if column.startswith('Ability - ')]
        ].mean(axis=1)

        self.survey['Perceived effectivity'] = self.survey[
                [column for column in self.survey.columns if column.startswith('Effectivity - ')]
        ].mean(axis=1)
        self.survey["How large is the area you grow crops on in hectares?"] = self.survey["How large is the area you grow crops on in acres?"] * 0.404686
        
        self.samples = self.survey[[
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

        for question, values in self.bins.items():
            assert len(values['bins']) == len(values['labels']) + 1, "Bin bounds must be one longer than labels"
            self.samples[question] = pd.cut(self.samples[question], bins=values['bins'], labels=values['labels'])

        # remove where data is -1
        self.samples = self.samples[(self.samples != -1).all(1)]
        # remove where data is NaN
        self.samples = self.samples.dropna()

        self.samples = self.samples.rename(columns=self.renames)

        self.fix_naming()

        return self.samples

class IHDSSurvey(Survey):
    def __init__(self):
        self.bins = {
            'age': {
                'bins': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                'labels': ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '>90'],
            },
            "area owned & cultivated (hectare)" : {
                'bins': [0, 1, 2, 4, np.inf],
                'labels': ['0-1', '1-2', '2-4', '>4'],  # after size class groups in agricultural census
            },
            'Education': {
                'bins': [-np.inf, 0, 5, 10, 12, np.inf],
                'labels': ['none', '5th_standard', 'matriculation', 'higher_secondary', 'graduate']
            },
            'Monthly consumption per capita Rs': {
                'bins': [50, 100, 250, 500, 1000, 2000, 3000, 4000, 5000, 10000],
                'labels': ['50-100', '100-250', '250-500', '500-1000', '1000-2000', '2000-3000', '3000-4000', '4000-5000', '5000-10000'],
            }
        }
        self.renames = {
            'age': 'age',
            'Education': 'education',
            'area owned & cultivated (hectare)': 'field_size',
            'Monthly consumption per capita Rs': 'monthly_consumption_per_capita',
        }
        self.survey = self.load()

    def load(self):
        df = pd.read_csv(PREPROCESSING_FOLDER / 'agents' / 'farmers' / 'IHDS_I.csv')
        return df

    def parse(self):
        self.samples = self.survey[['age', 'Education', 'area owned & cultivated (hectare)', 'Monthly consumption per capita Rs']]
        self.samples = self.samples[self.samples['Monthly consumption per capita Rs'] > 50]  # below 50 is unrealisicaly low
        self.samples = self.samples[self.samples['Monthly consumption per capita Rs'] < 10000]  # too little very rich people to be representative. Need specific study to adequately say something about them
        # drop all samples where at least one value is negative
        self.samples = self.samples[(self.samples >= 0).all(1)]
        for question, values in self.bins.items():
            assert len(values['bins']) == len(values['labels']) + 1, "Bin bounds must be one longer than labels"
            self.samples[question] = pd.cut(self.samples[question], bins=values['bins'], labels=values['labels'])
        self.samples = self.samples.rename(columns=self.renames)
        self.fix_naming()
        return self.samples

def sample(model, n, evidence=[]):
    print("Generating samples")
    sampler = BayesianModelSampling(model)
    # if no evidence this is equalivalent to forward sampling
    sample = sampler.rejection_sample(evidence=evidence, size=n, show_progress=True)
    return sample

def load(path):
    print("Loading model")
    return BayesianNetwork.load(str(path), filetype='bif')


if __name__ == '__main__':
    bayesian_net_folder = PREPROCESSING_FOLDER / 'bayesian_net'
    bayesian_net_folder.mkdir(exist_ok=True, parents=True)

    IHDS_survey = IHDSSurvey()
    IHDS_survey.parse()
    IHDS_survey.learn_structure()
    IHDS_survey.estimate_parameters(plot=False, save=bayesian_net_folder / 'IHDS.png')
    IHDS_survey.save(bayesian_net_folder / 'IHDS.bif')

    farmer_survey = FarmerSurvey(b'2!hM0t$2Kd66')
    farmer_survey.parse()
    farmer_survey.learn_structure()
    farmer_survey.estimate_parameters(plot=False, save=bayesian_net_folder / 'farmer_survey.png')
    farmer_survey.save(bayesian_net_folder / 'farmer_survey.bif')

    farmers = farmer_survey.sample(10)

    evidence_columns = ['age', 'education', 'field_size']

    def get_additional_variables(group):
        additional_variables = IHDS_survey.sample(
            n=len(group),
            evidence=tuple(group.iloc[0][evidence_columns]), # all rows have the same evidence
            evidence_columns=evidence_columns,
        )
        group['monthly_consumption_per_capita'] = additional_variables['monthly_consumption_per_capita'].tolist()  # if not converted to list pandas uses indices to match rows
        return group

    farmers = farmers.groupby(evidence_columns).apply(get_additional_variables)

    farmers.to_excel(PREPROCESSING_FOLDER / 'sample.xlsx')

    print(farmers)
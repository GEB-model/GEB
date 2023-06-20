from isimip_client.client import ISIMIPClient
client = ISIMIPClient()

# To mask a specific dataset, the following code can be used to first search the
# repository and then mask every file in the dataset. After starting the masking job 
# (on the server) it checks every 10 seconds, if the job is done and then downloads
# the file. Masking the files on the server can take a few minutes or even longer.

climate_variables = ['pr', 'rsds', 'tas', 'tasmax', 'tasmin']

# get the dataset metadata from the ISIMIP repository
response = client.datasets(simulation_round='ISIMIP3a',
                           product='InputData',
                           climate_forcing='chelsa-w5e5v1.0',
                           climate_scenario='obsclim',
                           climate_variable='tas',
                           resolution='30arcsec')

assert len(response["results"]) == 1
dataset = response["results"][0]
paths = [file['path'] for file in dataset['files']]

# start/poll a masking job on the server (e.g. for China)
response = client.cutout(paths, [-45.108, -41.935, 167.596, 173.644], poll=10)

# download the file when it is ready
client.download(response['file_url'], path='downloads', validate=False, extract=True)
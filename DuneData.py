import os
import pandas as pd
from duneanalytics import DuneAnalytics
from datetime import datetime as dt

def get_dune_data(qry_id):
    
	# Dune Login Info
	username = os.environ.get('dune_username')
	password = os.environ.get('dune_password')

	dune = DuneAnalytics(username,password)
	dune.login()
	dune.fetch_auth_token()

	result_id = dune.query_result_id_v3(query_id=qry_id)
	result = dune.get_execution_result(result_id)

	dune_data = pd.DataFrame(result['data']['get_execution']['execution_succeeded']['data'])

	return dune_data

def create_time_series(dune_data):

	dune_data['date'] = dune_data['block_date'].apply(lambda x: dt.strptime(x,'%Y-%m-%d %H:%M:%S.%f %Z'))
	cleaned_df = dune_data[['blockchain','token_pair','date','amount_usd']] \
		.groupby(['blockchain','token_pair','date']) \
			.agg(trades=('amount_usd', 'count')) \
				.reset_index()[['blockchain','token_pair','trades']] \
					.set_index(['blockchain','token_pair'])

	chains = cleaned_df.index.unique('blockchain').to_list()
	pairs = cleaned_df.index.unique('token_pair').to_list()

	trades_ts = {}
	for i in chains:
		chain = {}
		for j in pairs:
			ts = {j:cleaned_df.loc[i,j]['trades'].values.tolist()}
			chain.update(ts)
		blk = {i:chain}

	trades_ts.update(blk)

	return trades_ts
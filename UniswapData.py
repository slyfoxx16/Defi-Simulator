import requests
import json
import pandas as pd
import threading
import pickle
from datetime import datetime, timedelta

# Helper Functions
def price_calculator(amount0, amount1):
    price = abs(amount0) / abs(amount1)
    return price


def load_data_from_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# Uniswap Data Fetcher
class UniswapDataFetcher:
    """This is a class named UniswapDataFetcher which fetches the Uniswap data 
    related to the swap and pool daily data for multiple pools."""

    def __init__(self, pools, n_days):
        self.pools = pools
        self.n_days = n_days
        self.start_timestamp = int((datetime.utcnow() - timedelta(days=self.n_days)).timestamp())
        self.fetch_all_data()

    def display_fetch_info(self):
        pool_names = ", ".join(self.pools.keys())
        info_message = f"Fetching data for pools: **{pool_names}** for the last **{self.n_days}** days."
        print(info_message)

    def fetch_daily_data(self, pool_name, pool_id):
        """This method takes two arguments, the pool ID and the name of the pool, 
        and then creates a query to fetch the daily pool data from Uniswap V3 subgraph API 
        using a GraphQL query. After parsing the response, 
        it extracts the required information from it, 
        constructs a pandas dataframe out of it, sorts it by date, 
        and finally stores it in the daily_data attribute of the respective pool."""

        query = """
        query GetPoolDayData($id: String!, $startTimestamp: Int!) {
            poolDayDatas(
                where: { pool: $id, date_gt: $startTimestamp }
                orderBy: date, orderDirection: desc
            ) {
                date
                txCount
                volumeUSD
                liquidity
                feesUSD
                sqrtPrice
                tick
            }
        }
        """

        url = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
        variables = {"id": pool_id, "startTimestamp": self.start_timestamp}
        response = requests.post(url, json={"query": query, "variables": variables})

        try:
            daily_data = response.json()["data"]["poolDayDatas"]
        except KeyError:
            print(f"Error fetching daily data for {pool_name}: {response.text}")
            return

        date_list = []
        trading_volume_list = []
        liquidity_list = []
        txCount_list = []
        feesUSD_list = []
        sqrtPrice_list = []
        tick_list = []

        for day_data in daily_data:
            date = datetime.utcfromtimestamp(int(day_data["date"])).strftime("%Y-%m-%d")
            trading_volume = float(day_data["volumeUSD"])
            liquidity = float(day_data["liquidity"])
            txCount = float(day_data["txCount"])
            feesUSD = float(day_data["feesUSD"])
            sqrtPrice = float(day_data["sqrtPrice"])
            tick = float(day_data["tick"])

            date_list.append(date)
            trading_volume_list.append(trading_volume)
            liquidity_list.append(liquidity)
            txCount_list.append(txCount)
            feesUSD_list.append(feesUSD)
            sqrtPrice_list.append(sqrtPrice)
            tick_list.append(tick)

        df = pd.DataFrame({
            "Date": date_list,
            "Trading Volume": trading_volume_list,
            "Liquidity": liquidity_list,
            "Tx Count": txCount_list,
            "Fees USD": feesUSD_list,
            "Sqrt Price": sqrtPrice_list,
            "Tick": tick_list
            })
        
        # Organize data
        df['Date'] = df['Date'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
        df = df.set_index('Date').sort_index(ascending=True)

        self.pools[pool_name]['daily_data'] = df.sort_index()

    def fetch_swap_data(self, pool_name, pool_id):
        """This method also takes two arguments, the pool ID and the name of the pool, 
        and then creates a query to fetch the swap data from Uniswap V3 subgraph API 
        using a GraphQL query. After parsing the response, 
        it extracts the required information from it, 
        constructs a pandas dataframe out of it, 
        and finally stores it in the swap_data attribute of the respective pool."""

        query = """
        query GetSwaps($id: String!, $startTimestamp: Int!, $skip: Int!) {
            swaps(
                where: { pool: $id, timestamp_gt: $startTimestamp }
                orderBy: timestamp, orderDirection: desc
                skip: $skip
            ) {
                id
                timestamp
                pool
                token0 { symbol }
                token1 { symbol }
                tick
                amount0
                amount1
                amountUSD
                sqrtPriceX96
            }
        }
        """

        url = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
        swap_data = []
        skip = 0
        while True:
            variables = {"id": pool_id, "startTimestamp": self.start_timestamp, "skip": skip}
            response = requests.post(url, json={"query": query, "variables": variables})

            if response.status_code != 200:
                print(f"Error fetching data for {pool_name}: {response.text}")
                return None

            new_data = json.loads(response.text)["data"]["swaps"]
            if not new_data:
                break

            swap_data.extend(new_data)
            skip += len(new_data)

        df = pd.DataFrame(swap_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df["token0"] = df["token0"].apply(lambda x: x["symbol"])
        df["token1"] = df["token1"].apply(lambda x: x["symbol"])

        # Cast strings to numeric
        df["tick"] = df["tick"].astype(int)
        df[['amount0', 'amount1', 'sqrtPriceX96', 'amountUSD']] = df[['amount0', 'amount1', 'sqrtPriceX96', 'amountUSD']] \
            .astype(float)

        # Calculate price ratio and prices in both directions
        df["price"] = df.apply(lambda row: abs(row["amount0"]) / abs(row["amount1"]),axis=1)

        self.pools[pool_name]['swap_data'] = df.set_index("timestamp")

    def fetch_all_data(self):
        """This method starts threads for both of the above methods for all given pools 
        and waits for them to finish using the join() method 
        on each thread object in the respective list of threads. 
        This ensures that all threads are complete before returning the final output."""

        self.display_fetch_info()

        swap_threads = []
        pool_day_threads = []

        for pool_name in self.pools.keys():
            swap_thread = threading.Thread(target=self.fetch_swap_data, args=(pool_name,self.pools[pool_name]['address']))
            swap_thread.start()
            swap_threads.append(swap_thread)

            pool_day_thread = threading.Thread(target=self.fetch_daily_data, args=(pool_name,self.pools[pool_name]['address']))
            pool_day_thread.start()
            pool_day_threads.append(pool_day_thread)

        for thread in swap_threads + pool_day_threads:
            thread.join()

    def save_data_to_pickle(self):
        now = datetime.now().strftime("%Y%m%d")
        filename = 'uniswap_data_' + now + '_' + str(self.n_days) + 'days.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self.pools, f)
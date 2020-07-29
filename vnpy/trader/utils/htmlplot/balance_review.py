from htmlplot.core import MultiPlot
from queue import Queue
import os
import pandas as pd
from datetime import datetime, timedelta
from pymongo import MongoClient
import numpy as np


def db_get_kline(symbol: str, start: datetime, end: datetime, host: str="localhost", db: str="Kline_1Min_Auto_Db_Plus"):
    ft = {
        "datetime": {
            "$gte": start,
            "$lte": end
        }
    }
    docs = list(MongoClient(host)[db][symbol].find(ft, projection=["datetime", "open", "high", "low", "close"]))
    return pd.DataFrame(docs)


def make_plot(kline: pd.DataFrame, trades: pd.DataFrame, holding: pd.DataFrame):
    mp = MultiPlot()
    mp.set_main(kline, trades)
    mp.set_vbar(holding, colors={"long": "green", "short": "red"})
    mp.show()


SIDE_MAP = {"buy": 1, "sell": -1}
POS_MAP = {"open": 1, "close": -1}


def read_orders(filename: str):
    data = pd.read_excel(filename)
    data["Datetime"] = [datetime.fromisoformat(dt) for dt in data["Datetime"]]
    data["direction"] = data["Side"].apply(lambda s: SIDE_MAP[s]) * data["Position"].apply(lambda p: POS_MAP[p])
    return data.sort_values("Datetime")


def make_trade(data: pd.DataFrame):
    trades = []
    for extra, frame in data.groupby(data["Extra"]):
        long_trade = list(iter_trade(frame[frame.direction==1], extra))
        short_trade = list(iter_trade(frame[frame.direction==-1], extra))
        trades.extend(long_trade)
        trades.extend(short_trade)
    return pd.DataFrame(trades).sort_values("exitDt")


def cal_position(data: pd.DataFrame):
    side = data["Side"].apply(lambda s: SIDE_MAP[s])
    amount = data["Fill_Qty"] * side
    position = pd.DataFrame({"datetime": data["Datetime"], "amount": amount})
    position["holding"] = position["amount"].cumsum()
    return position.set_index("datetime")


def iter_holding_cost(trades: pd.DataFrame):
    for trade in trades.to_dict("record"):
        yield {"datetime": trade["entryDt"], "cost": trade["entryPrice"]*trade["volume"], "volume": trade["volume"]}
        yield {"datetime": trade["exitDt"], "cost": -trade["entryPrice"]*trade["volume"], "volume": -trade["volume"]}


def cal_holding_cost(trades: pd.DataFrame):
    cost = pd.DataFrame(list(iter_holding_cost(trades))).sort_values("datetime").set_index("datetime")
    cost["holding"] = cost["volume"].cumsum()
    cost["amount"] = cost["cost"].cumsum() 
    cost["avg_holding_price"] = cost["amount"] / cost["holding"].replace(0, np.NaN)
    return cost


def expand(data: pd.Series, index: list, freq: timedelta=timedelta(minutes=1)):
    origin_index = [dt - timedelta(seconds=dt.timestamp() % freq.total_seconds()) for dt in data.index]
    array = data.groupby(origin_index).last()
    joined_index = set(index)
    joined_index.update(origin_index)
    return pd.Series(array, list(sorted(joined_index))).ffill().dropna()


def iter_trade(data: pd.DataFrame, extra: str):
    data = data[(data["Position"].apply(lambda p: POS_MAP[p]) * data["Fill_Qty"]).cumsum() >= 0]
    q = Queue()
    front = None
    for doc in data.to_dict("record"):
        if doc["Position"] == "open":
            q.put(doc)
        elif doc["Position"] == "close":
            if not front:
                front = q.get()
            
            while doc["Fill_Qty"] > front["Fill_Qty"]:
                trade = {
                    "entryDt": front["Datetime"],
                    "entryPrice": front["Price"],
                    "exitDt": doc["Datetime"],
                    "exitPrice": doc["Price"],
                    "volume": front["Fill_Qty"] * doc["direction"],
                    "extra": extra
                }
                yield trade
                doc["Fill_Qty"] -= front["Fill_Qty"]
                front = q.get()

            if doc["Fill_Qty"] <= front["Fill_Qty"]:
                trade = {
                    "entryDt": front["Datetime"],
                    "entryPrice": front["Price"],
                    "exitDt": doc["Datetime"],
                    "exitPrice": doc["Price"],
                    "volume": doc["Fill_Qty"] * doc["direction"],
                    "extra": extra
                }
                yield trade
                front["Fill_Qty"] -= doc["Fill_Qty"]
                if front["Fill_Qty"] == 0:
                    front = None


class BalanceTemplate(object):

    def __init__(self):
        self.orders: pd.DataFrame = None
        self.trades: pd.DataFrame = None
        self.kline: pd.DataFrame = None
        self.freq: timedelta = timedelta(minutes=1)
    
    def assert_properties(self):
        assert isinstance(self.orders, pd.DataFrame)
        assert len(self.orders)
        assert isinstance(self.kline, pd.DataFrame)
        assert len(self.kline)
    
    def show(self):
        self.assert_properties()
        if not isinstance(self.trades, pd.DataFrame):
            self.trades = make_trade(self.orders)
        trades = self.trades
        orders = self.orders
        idx = list(self.kline["datetime"])
        cost_dict = {}
        for key, value in [("long", trades[trades["volume"]>0]), ("short", trades[trades["volume"]<0])]:
            if len(value):
                cost_dict[key] = expand(cal_holding_cost(value)["avg_holding_price"], idx)
        cost = pd.DataFrame(cost_dict)
        hold_dict = {}
        for key, value in [("long", orders[orders.direction==1]), ("short", orders[orders.direction==-1])]:
            if len(value):
                hold_dict[key] = expand(cal_position(value)["holding"], idx)
        holding = pd.DataFrame(hold_dict)

        mp = MultiPlot()
        mp.set_main(self.kline, trades)
        pos = mp.set_vbar(holding, colors={"long": "green", "short": "red"})
        mp.updateConfig(pos, {"plot_height": 250, "title": "holding"})
        pos = mp.set_line(cost, colors={"long": "green", "short": "red"})
        mp.updateConfig(pos, {"plot_height": 250, "title": "avg_holding_price"})
        mp.show()




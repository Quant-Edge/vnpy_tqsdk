from datetime import timedelta
from collections.abc import Callable
import traceback

import pandas as pd
from loguru import logger
from tqdm import tqdm
from tqsdk import TqApi, TqAuth

from vnpy.trader.datafeed import BaseDatafeed
from vnpy.trader.setting import SETTINGS
from vnpy.trader.constant import Interval
from vnpy.trader.object import BarData, HistoryRequest
from vnpy.trader.utility import ZoneInfo


INTERVAL_VT2TQ: dict[Interval, int] = {
    Interval.MINUTE: 60,
    Interval.HOUR: 60 * 60,
    Interval.DAILY: 60 * 60 * 24,
}

CHINA_TZ = ZoneInfo("Asia/Shanghai")


class TqsdkDatafeed(BaseDatafeed):
    """天勤TQsdk数据服务接口"""

    def __init__(self):
        """"""
        self.username: str = SETTINGS["datafeed.username"]
        self.password: str = SETTINGS["datafeed.password"]

    def query_bar_history(
        self, req: HistoryRequest, output: Callable = print
    ) -> list[BarData] | None:
        """查询k线数据"""
        # 初始化API
        try:
            api: TqApi = TqApi(auth=TqAuth(self.username, self.password))
        except Exception:
            output(traceback.format_exc())
            return None

        # 查询数据
        interval: str = INTERVAL_VT2TQ.get(req.interval, None)
        if not interval:
            output(f"Tqsdk查询K线数据失败：不支持的时间周期{req.interval.value}")
            return []

        tq_symbol: str = f"{req.exchange.value}.{req.symbol}"

        df: pd.DataFrame = api.get_kline_data_series(
            symbol=tq_symbol,
            duration_seconds=interval,
            start_dt=req.start,
            end_dt=(req.end + timedelta(1)),
        )

        # 关闭API
        api.close()

        # 解析数据
        bars: list[BarData] = []

        if df is not None:
            for tp in df.itertuples():
                # 天勤时间为与1970年北京时间相差的秒数，需要加上8小时差
                dt: pd.Timestamp = pd.Timestamp(
                    tp.datetime
                ).to_pydatetime() + timedelta(hours=8)

                bar: BarData = BarData(
                    symbol=req.symbol,
                    exchange=req.exchange,
                    interval=req.interval,
                    datetime=dt.replace(tzinfo=CHINA_TZ),
                    open_price=tp.open,
                    high_price=tp.high,
                    low_price=tp.low,
                    close_price=tp.close,
                    volume=tp.volume,
                    open_interest=tp.open_oi,
                    gateway_name="TQ",
                )
                bars.append(bar)

        return bars

    def query_all(
        self, ind_class: str, interval: Interval, data_length: int = 1e4
    ) -> pd.DataFrame | None:
        """查询k线数据"""
        # 初始化API
        duration_seconds: str = INTERVAL_VT2TQ.get(interval, None)
        r_df_list = []
        with TqApi(auth=TqAuth(self.username, self.password)) as api:
            symbols = api.query_quotes(ins_class=ind_class)
            for symbol in tqdm(symbols):
                r = api.get_kline_serial(
                    symbol=symbol,
                    duration_seconds=duration_seconds,
                    data_length=data_length,
                ).dropna()
                logger.info(f"[{ind_class}] {symbol} {r.shape} / {data_length}")
                r_df_list.append(r)

        if r_df_list:
            df = pd.concat(r_df_list).drop(['id', 'duration'], axis=1)
        else:
            df = None

        return df

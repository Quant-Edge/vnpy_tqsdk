import concurrent.futures
import datetime as dt
from collections.abc import Callable
from pathlib import Path
import traceback

import pandas as pd
import polars as pl
from loguru import logger
from tqdm import tqdm
from tqsdk import TqApi, TqAuth

from vnpy.trader.datafeed import BaseDatafeed
from vnpy.trader.setting import SETTINGS
from vnpy.trader.constant import Interval
from vnpy.trader.object import BarData, HistoryRequest
from vnpy.trader.utility import ZoneInfo
from vnpy.trader.database import DB_TZ


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

        self.drop_cols = ["id", "duration"]

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
            end_dt=(req.end + dt.timedelta(1)),
        )

        # 关闭API
        api.close()

        # 解析数据
        bars: list[BarData] = []

        if df is not None:
            for tp in df.itertuples():
                # 天勤时间为与1970年北京时间相差的秒数，需要加上8小时差
                datetime: pd.Timestamp = pd.Timestamp(
                    tp.datetime
                ).to_pydatetime() + dt.timedelta(hours=8)

                bar: BarData = BarData(
                    symbol=req.symbol,
                    exchange=req.exchange,
                    interval=req.interval,
                    datetime=datetime.replace(tzinfo=CHINA_TZ),
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

    def query_free_all(
        self, ind_class: str, interval: Interval, data_length: int = 1e4
    ) -> pl.DataFrame | None:
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
            df = pd.concat(r_df_list).drop(self.drop_cols, axis=1)
            df = self.format_df(df)
        else:
            df = None

        return df

    def get_early_pro_data(
        self,
        cur_df: pl.DataFrame,
        interval: Interval,
        start_datetime: dt.datetime = dt.datetime(2010, 1, 1),
        adj_type: str | None = None,
    ) -> pl.DataFrame:
        early_df_list = []
        iter_df = cur_df.group_by("jj_code").agg(pl.col("open_time").min())

        def fetch_data(row):
            jj_code = row["jj_code"]
            end = row["open_time"] - self.get_time_step(interval)
            duration_seconds = INTERVAL_VT2TQ.get(interval, None)
            # 每个线程单独创建 TqApi 实例
            with TqApi(auth=TqAuth(self.username, self.password)) as api:
                try:
                    df = api.get_kline_data_series(
                        symbol=jj_code,
                        duration_seconds=duration_seconds,
                        start_dt=start_datetime,
                        end_dt=end,
                        adj_type=adj_type,
                    )
                    before_drop_shape = df.shape
                    df = df.dropna()
                    logger.info(f"{jj_code} {before_drop_shape} -> {df.shape}")
                except Exception as e:
                    logger.error(f"{jj_code} {e}")
                    return None

            if df.empty:
                logger.warning(f"{jj_code} no early data.")
                return None
            df = df.drop(self.drop_cols, axis=1)
            return self.format_df(df)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(fetch_data, iter_df.iter_rows(named=True)), total=len(iter_df), desc=f"Total {len(iter_df)}"))

        early_df_list = [result for result in results if result is not None]

        return pl.concat(early_df_list) if early_df_list else pl.DataFrame()

    def query_pro_data(
        self,
        symbol: str,
        interval: Interval,
        start: dt.datetime,
        end: dt.datetime,
        adj_type: str | None = None,
    ) -> pd.DataFrame:
        """
        查询pro数据
        """
        duration_seconds: str = INTERVAL_VT2TQ.get(interval, None)
        with TqApi(auth=TqAuth(self.username, self.password)) as api:
            df = api.get_kline_data_series(
                symbol=symbol,
                duration_seconds=duration_seconds,
                start_dt=start,
                end_dt=end,
                adj_type=adj_type,
            ).dropna()
            if not df.empty:
                df = df.drop(self.drop_cols, axis=1)

        return df

    @staticmethod
    def format_df(df: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
        if isinstance(df, pd.DataFrame):
            df = pl.from_pandas(df)
        return (
            df.with_columns(
                pl.col("datetime").cast(pl.Datetime(time_unit="ns", time_zone=DB_TZ))
            )
            .rename({"datetime": "open_time", "symbol": "jj_code"})
            .with_columns(
                close_time=pl.col("open_time")
                + pl.duration(minutes=59, seconds=59, milliseconds=999)
            )
        )

    @staticmethod
    def get_time_step(interval: Interval) -> dt.timedelta:
        return {
            Interval.MINUTE: dt.timedelta(minutes=1),
            Interval.HOUR: dt.timedelta(hours=1),
            Interval.DAILY: dt.timedelta(days=1),
        }[interval]

    @staticmethod
    def save_df(df: pd.DataFrame | pl.DataFrame, file_path: Path):
        if isinstance(df, pd.DataFrame):
            df.to_parquet(file_path, index=False)
        elif isinstance(df, pl.DataFrame):
            df.write_parquet(file_path)
        else:
            raise TypeError(f"Unsupported type: {type(df)}")

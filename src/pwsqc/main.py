import pandas as pd
from pwsqc import apply_flags
from pwsqc import bias_correction
from pwsqc import Config
from pwsqc import faulty_zero_filter
from pwsqc import find_station_neighbors
from pwsqc import high_influx_filter
from pwsqc import prepare_timeseries
from pwsqc import station_outlier_filter


def main() -> int:
    config = Config()

    df_data = pd.read_parquet('../data/precip/Valencia/station_data_Valencia.parquet')
    df_data = prepare_timeseries(df_data)
    df_meta = pd.read_parquet(
        '../data/precip/Valencia/station_metadata_Valencia.parquet',
    )
    neighbors = find_station_neighbors(df_meta, d=config.d)

    df_data = faulty_zero_filter(
        df_data,
        neighbors=neighbors,
        n_stat=config.n_stat,
        n_int=config.n_int,
    )
    df_data = high_influx_filter(
        df_data,
        neighbors=neighbors,
        n_stat=config.n_stat,
        phi_a=config.phi_a,
        phi_b=config.phi_b,
    )
    df_data = station_outlier_filter(
        df_data,
        neighbors=neighbors,
        n_stat=config.n_stat,
        m_int=config.m_int,
        m_rain=config.m_rain,
        m_match=config.m_match,
        gamma=config.gamma,
        dbc=config.dbc,
    )
    df_data = bias_correction(df_data, dbc=config.dbc, beta=config.beta)
    # discard the flagged intervals and apply the bias correction, pass
    # strict=True to also discard the intervals no flag could be attributed to
    df_data = apply_flags(df_data, strict=False)

    df_data.to_parquet('../data/precip/Valencia/station_data_Valencia_qc.parquet')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

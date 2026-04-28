from unittest import mock

import pytest

import wrds


def test_rawsql_pandas_takes_unparameterized_sql(mock_connection):
    """Test raw_sql handles unparameterized SQL queries with the pandas backend."""
    with mock.patch("wrds.sql.sa"):
        with mock.patch("wrds.sql.POLARS", False):
            with mock.patch("wrds.sql.dfp") as mock_dfp:
                mock_connection.connection = mock.Mock()
                mock_connection.engine = mock.Mock()
                sql = "SELECT * FROM information_schema.tables LIMIT 1"
                mock_connection.raw_sql(sql)
                mock_dfp.read_sql_query.assert_called_once_with(
                    sql,
                    mock_connection.connection,
                    coerce_float=True,
                    parse_dates=None,
                    index_col=None,
                    chunksize=500000,
                    params=None,
                    dtype=None,
                    dtype_backend="numpy_nullable",
                )


def test_rawsql_pandas_takes_parameterized_sql(mock_connection):
    """Test raw_sql handles parameterized SQL queries with the pandas backend."""
    with mock.patch("wrds.sql.sa"):
        with mock.patch("wrds.sql.POLARS", False):
            with mock.patch("wrds.sql.dfp") as mock_dfp:
                mock_connection.connection = mock.Mock()
                mock_connection.engine = mock.Mock()
                sql = (
                    "SELECT * FROM information_schema.tables "
                    "WHERE table_name = %(tablename)s LIMIT 1"
                )
                tablename = "pg_stat_activity"
                mock_connection.raw_sql(sql, params=tablename)
                mock_dfp.read_sql_query.assert_called_once_with(
                    sql,
                    mock_connection.connection,
                    coerce_float=True,
                    parse_dates=None,
                    index_col=None,
                    chunksize=500000,
                    params=tablename,
                    dtype=None,
                    dtype_backend="numpy_nullable",
                )


def test_rawsql_polars_takes_unparameterized_sql(mock_connection):
    """Test raw_sql handles unparameterized SQL queries with the polars backend."""
    with mock.patch("wrds.sql.sa"):
        with mock.patch("wrds.sql.POLARS", True):
            with mock.patch("wrds.sql.dfp") as mock_dfp:
                mock_connection.connection = mock.Mock()
                mock_connection.engine = mock.Mock()
                sql = "SELECT * FROM information_schema.tables LIMIT 1"
                mock_connection.raw_sql(sql)
                mock_dfp.read_database.assert_called_once_with(
                    sql,
                    mock_connection.connection,
                    iter_batches=False,
                    batch_size=None,
                    schema_overrides=None,
                    execute_options=None,
                )


def test_rawsql_polars_takes_parameterized_sql(mock_connection):
    """Test raw_sql handles parameterized SQL queries with the polars backend."""
    with mock.patch("wrds.sql.sa"):
        with mock.patch("wrds.sql.POLARS", True):
            with mock.patch("wrds.sql.dfp") as mock_dfp:
                mock_connection.connection = mock.Mock()
                mock_connection.engine = mock.Mock()
                sql = (
                    "SELECT * FROM information_schema.tables "
                    "WHERE table_name = %(tablename)s LIMIT 1"
                )
                tablename = "pg_stat_activity"
                mock_connection.raw_sql(sql, params=tablename)
                mock_dfp.read_database.assert_called_once_with(
                    sql,
                    mock_connection.connection,
                    iter_batches=False,
                    batch_size=None,
                    schema_overrides=None,
                    execute_options={"parameters": tablename},
                )


def test_rawsql_polars_return_iter(mock_connection):
    """Test raw_sql passes iter_batches and batch_size when return_iter=True."""
    with mock.patch("wrds.sql.sa"):
        with mock.patch("wrds.sql.POLARS", True):
            with mock.patch("wrds.sql.dfp") as mock_dfp:
                mock_connection.connection = mock.Mock()
                mock_connection.engine = mock.Mock()
                sql = "SELECT * FROM information_schema.tables LIMIT 1"
                mock_connection.raw_sql(sql, return_iter=True, chunksize=1000)
                mock_dfp.read_database.assert_called_once_with(
                    sql,
                    mock_connection.connection,
                    iter_batches=True,
                    batch_size=1000,
                    schema_overrides=None,
                    execute_options=None,
                )


@pytest.mark.skipif(not wrds.sql.POLARS, reason="polars not installed")
def test_rawsql_polars_is_default_when_available():
    """Test that POLARS=True when polars is installed."""
    assert wrds.sql.POLARS is True

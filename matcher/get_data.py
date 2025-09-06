import duckdb


OS_PARQUET = "/Users/robin.linacre/Documents/data_linking/uk_address_matcher/secret_data/ord_surv/raw/add_gb_builtaddress_sorted_zstd.parquet"
FHRS_PATH = "/Users/robin.linacre/Documents/data_linking/uk_address_matcher/example_data/fhrs_addresses_sample.parquet"


CLEAN_PIPELINE_SQL = """
                .upper()
                .regexp_replace('[,.]', ' ', 'g')
                .regexp_replace('\\s+', ' ', 'g')
                .trim()
                .str_split(' ')
"""


PC_REMOVE_TMPL_SQL = ".regexp_replace('{pc}', '', 'gi')"


def get_random_address_data(
    postcode: str | None = None,
    fhrs_path: str = FHRS_PATH,
    os_parquet_path: str = OS_PARQUET,
    connection: duckdb.DuckDBPyConnection | None = None,
    print_output=False,
):
    con = connection or duckdb.connect(":default:")

    def esc(s: str) -> str:
        # escape single quotes for safe SQL string literal embedding
        return s.replace("'", "''")

    # 1) Determine the postcode to use
    if postcode is None or not str(postcode).strip():
        pc_sql = f"""
            SELECT postcode AS pc
            FROM read_parquet('{esc(fhrs_path)}')
            WHERE postcode IS NOT NULL AND LENGTH(TRIM(postcode)) > 0
            ORDER BY random()
            LIMIT 1
        """
        postcode = con.execute(pc_sql).fetchone()[0]

    pc = str(postcode).strip().upper()
    pc_short = pc[:-1]  # remove last char

    # 2) Build relations

    # FHRS: one row for the exact postcode, with tokenized address_concat
    fhrs_sql = f"""
        SELECT
            unique_id,
            address_concat{CLEAN_PIPELINE_SQL} AS tokens,
            postcode
        FROM read_parquet('{esc(fhrs_path)}')
        WHERE postcode = '{esc(pc)}'
        LIMIT 1
    """
    fhrs_rel = con.sql(fhrs_sql)

    # OS: all rows whose postcode shares the prefix (postcode minus last char),
    # tokenizing fulladdress after removing the full postcode text
    os_sql = f"""
        SELECT
            uprn,
            fulladdress{PC_REMOVE_TMPL_SQL.format(pc=esc(pc))}{CLEAN_PIPELINE_SQL} AS tokens,
            postcode
        FROM read_parquet('{esc(os_parquet_path)}')
        WHERE LEFT(postcode, LENGTH(postcode)-1) = '{esc(pc_short)}'
        order by postcode
    """
    os_rel = con.sql(os_sql)

    if print_output:
        fhrs_rel.show(max_width=20000)
        os_rel.show(max_width=20000)

    return fhrs_rel.fetchall()[0], os_rel.fetchall()


def get_address_data_from_messy_address(
    address_no_postcode: str,
    postcode: str,
    os_parquet_path: str = OS_PARQUET,
    connection: duckdb.DuckDBPyConnection | None = None,
    print_output=False,
):
    con = connection or duckdb.connect(":default:")

    def esc(s: str) -> str:
        return s.replace("'", "''")

    pc = str(postcode).strip().upper()
    pc_short = pc[:-1]  # remove last char

    # 1) Clean/tokenize the input messy address using the same FHRS rules
    input_sql = f"""
        SELECT
            CAST(1 AS BIGINT) AS unique_id,
            addr{CLEAN_PIPELINE_SQL} AS tokens,
            postcode
        FROM (SELECT '{esc(address_no_postcode)}' AS addr, '{esc(pc)}' AS postcode)
    """
    input_rel = con.sql(input_sql)

    # 2) OS rows for the postcode prefix, tokenizing fulladdress with the postcode removed
    os_sql = f"""
        SELECT
            uprn,
            fulladdress{PC_REMOVE_TMPL_SQL.format(pc=esc(pc))}{CLEAN_PIPELINE_SQL} AS tokens,
            postcode
        FROM read_parquet('{esc(os_parquet_path)}')
        WHERE LEFT(postcode, LENGTH(postcode)-1) = '{esc(pc_short)}'
        order by postcode
    """
    os_rel = con.sql(os_sql)

    if print_output:
        input_rel.show(max_width=20000)
        os_rel.show(max_width=20000)

    return input_rel.fetchone(), os_rel.fetchall()

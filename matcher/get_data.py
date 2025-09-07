import duckdb


OS_PARQUET = "/Users/robin.linacre/Documents/data_linking/uk_address_matcher/secret_data/ord_surv/raw/add_gb_builtaddress_sorted_zstd.parquet"
FHRS_PATH = "/Users/robin.linacre/Documents/data_linking/uk_address_matcher/secret_data/fhrs/fhrs_data.parquet"


CLEAN_PIPELINE_SQL = """
                .upper()
                -- replace commas, periods, and apostrophes with spaces
                .regexp_replace('[,.'']', ' ', 'g')
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
            fhrsid as unique_id,
            concat_ws(' ', businessname, addressline1, addressline2, addressline3, addressline4){CLEAN_PIPELINE_SQL} AS tokens,

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
        sql = f"""
        select *
          FROM read_parquet('{esc(fhrs_path)}')
          where fhrsid in (select unique_id from fhrs_rel)"""
        con.sql(sql).show(max_width=20000)

        sql = f"""
        select uprn, fulladdress
            FROM read_parquet('{esc(os_parquet_path)}')
            where postcode = '{esc(pc)}'"""
        con.sql(sql).show(max_width=20000)

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


def show_uprns(
    uprns,
    os_parquet_path: str = OS_PARQUET,
    connection: duckdb.DuckDBPyConnection | None = None,
    *,
    max_width: int = 20000,
) -> None:
    """Display OS fulladdress rows for the given UPRN list using DuckDB .show().

    Example:
        >>> show_uprns([2630107862])
    """
    try:
        vals = [int(u) for u in (uprns or [])]
    except Exception:
        vals = []
    if not vals:
        print("(no UPRNs to show)")
        return

    con = connection or duckdb.connect(":default:")

    def esc(s: str) -> str:
        return s.replace("'", "''")

    in_list = ",".join(str(u) for u in vals)
    sql = f"""
        SELECT uprn, fulladdress, postcode
        FROM read_parquet('{esc(os_parquet_path)}')
        WHERE uprn IN ({in_list})
        ORDER BY uprn
    """
    con.sql(sql).show(max_width=max_width)


def show_postcode(
    postcode: str,
    os_parquet_path: str = OS_PARQUET,
    connection: duckdb.DuckDBPyConnection | None = None,
    *,
    max_width: int = 20000,
) -> None:
    """Display all OS fulladdress rows for the given postcode.

    Example:
        >>> show_postcode("WD4 9HW")
    """
    pc = (postcode or "").strip()
    if not pc:
        print("(no postcode provided)")
        return
    con = connection or duckdb.connect(":default:")

    def esc(s: str) -> str:
        return s.replace("'", "''")

    sql = f"""
        SELECT uprn, fulladdress, postcode
        FROM read_parquet('{esc(os_parquet_path)}')
        WHERE postcode = '{esc(pc)}'
        ORDER BY fulladdress
    """
    con.sql(sql).show(max_width=max_width)


def show_postcode_by_levenshtein(
    messy_text: str,
    postcode: str,
    os_parquet_path: str = OS_PARQUET,
    connection: duckdb.DuckDBPyConnection | None = None,
    *,
    max_width: int = 20000,
) -> None:
    """Display all OS fulladdress rows for the given postcode, ordered by Levenshtein
    distance to the provided messy_text, after applying the same simple cleaning
    (uppercase, strip postcode, remove punctuation, collapse spaces).

    Example:
        >>> show_postcode_by_levenshtein("CORSBIE VILLA GUEST HOUSE CORSBIE ROAD NEWTON STEWART", "DG8 6JB")
    """
    pc = (postcode or "").strip()
    if not pc:
        print("(no postcode provided)")
        return
    txt = messy_text or ""
    con = connection or duckdb.connect(":default:")

    def esc(s: str) -> str:
        return s.replace("'", "''")

    # Clean messy text inline in SQL to mirror the address cleaning
    sql = f"""
        WITH src AS (
            SELECT uprn, fulladdress, postcode
            FROM read_parquet('{esc(os_parquet_path)}')
            WHERE postcode = '{esc(pc)}'
        ),
        params AS (
            SELECT '{esc(txt)}' AS messy_base
        ),
        cleaned AS (
            SELECT
                s.uprn,
                s.fulladdress,
                s.postcode,
                -- Clean canonical: remove exact postcode text then normalize
                (s.fulladdress{PC_REMOVE_TMPL_SQL.format(pc=esc(pc))}
                    .upper()
                    .regexp_replace('[,.'']', ' ', 'g')
                    .regexp_replace('\\s+', ' ', 'g')
                    .trim()
                ) AS canon_clean,
                -- Clean messy text similarly (from params CTE)
                (p.messy_base
                    .upper()
                    .regexp_replace('[,.'']', ' ', 'g')
                    .regexp_replace('\\s+', ' ', 'g')
                    .trim()
                ) AS messy_clean
            FROM src s, params p
        )
        SELECT uprn, fulladdress, postcode,
               levenshtein(messy_clean, canon_clean) AS dist,
               messy_clean AS messy_norm,
               canon_clean AS canon_norm
        FROM cleaned
        ORDER BY dist, fulladdress
    """

    con.sql(sql).show(max_width=max_width)

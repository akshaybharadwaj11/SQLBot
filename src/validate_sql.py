import sqlparse

def sanitize_sql(sql: str) -> str:
    # basic formatter, extend with allowlists/deny rules
    return sqlparse.format(sql, reindent=True, keyword_case="upper")
def remove_unneccessary_columns(lf):
    return lf.select(["Date", "Article_title", "Stock_symbol", "summary"])

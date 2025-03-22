import datetime as dt
from typing import Any, Dict, List, Optional, TypedDict

from mc_providers.provider import AllItems, Date, Item, Items

class SearchApiClient:
    TIMEOUT_SECS: int

    def __init__(self, collection: str, base_url: str): ...

    def all_articles(self, query: str, start_date: dt.datetime, end_date: dt.datetime,
                     page_size: int, **kwargs: Any) -> AllItems: ...

    def article(self, item_id: str) -> Item: ...

    def count(self, query: str, start_date: dt.datetime, end_date: dt.datetime,
              **kwargs: Any) -> int: ...

    def count_over_time(self, query: str, start_date: dt.datetime, end_date: dt.datetime,
                        **kwargs: Any) -> List[Date]: ...

    def sample(self, query: str,
               start_date: dt.datetime, end_date: dt.datetime, limit: int,
               **kwargs: Any) -> Items: ...


    def paged_articles(self, query: str, start_date: dt.datetime, end_date: dt.datetime,
                       page_size: Optional[int] = 1000,  expanded: bool = False,
                       pagination_token: Optional[str] = None, **kwargs: Any) -> tuple[List[Dict], Optional[str]]:
        ...

    def top_languages(self, query: str, start_date: dt.datetime, end_date: dt.datetime, **kwargs: Any) -> List[Dict]:
        ...

    def top_sources(self, query: str, start_date: dt.datetime, end_date: dt.datetime, **kwargs: Any) -> List[Dict]:
        ...

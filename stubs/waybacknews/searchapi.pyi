import datetime as dt
from typing import Any, TypedDict

from mc_providers.provider import AllItems, CountOverTime, Item, Items

class Match(TypedDict):
    """Wayback internal field names"""
    domain: str
    id: str
    title: str
    publish_date: str
    url: str
    language: str
    archive_playback_url: str
    article_url: str

class SearchApiClient:
    TIMEOUT_SECS: int

    def __init__(self, collection: str, base_url: str): ...

    def all_articles(self, query: str, start_date: dt.datetime, end_date: dt.datetime,
                     page_size: int, **kwargs: Any) -> AllItems: ...

    def article(self, item_id: str) -> Item: ...

    def count(self, query: str, start_date: dt.datetime, end_date: dt.datetime,
              **kwargs: Any) -> int: ...

    def count_over_time(self, query: str, start_date: dt.datetime, end_date: dt.datetime,
                        **kwargs: Any) -> CountOverTime: ...

    def sample(self, query: str,
               start_date: dt.datetime, end_date: dt.datetime, limit: int,
               **kwargs: Any) -> Items: ...


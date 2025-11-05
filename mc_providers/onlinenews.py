import datetime as dt
import json
import logging
import random
from collections import Counter
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, TypeAlias, TypedDict

# PyPI
import ciso8601
import dateparser     # used for publication_date in IA match_to_row
import numpy as np              # for chunking
from waybacknews.searchapi import SearchApiClient

from .provider import (
    AllItems, ContentProvider, CountOverTime, Date,
    Item, Items, Language, Source, Terms, Trace,
    LANGUAGES_LIMIT, SAMPLE_LIMIT,
    SOURCES_LIMIT, WORDS_LIMIT
)
from .cache import CachingManager


# don't need a logger per Provider instance
logger = logging.getLogger(__name__)

Counts: TypeAlias = dict[str, int]         # key: count
UrlSearchStrings: TypeAlias = Mapping[str, set[str]] # want Countable[str]

class Overview(TypedDict):
    query: str
    total: int
    topdomains: Counts          # from _format_counts
    toplangs: Counts            # from _format_counts
    dailycounts: Counts         # from _format_day_counts

SourcesByDate: TypeAlias = dict[str, Counts]

class OnlineNewsAbstractProvider(ContentProvider):
    """
    All these endpoints accept `domains: List[str]`
    and `filter: List[str] search keyword args.
    """

    MAX_QUERY_LENGTH = pow(2, 14)

    # default values for constructor arguments
    API_KEY = ""                # not required

    # no class-specific __init__

    @classmethod
    def domain_search_string(cls) -> str:
        raise NotImplementedError("Abstract provider class should not be implemented directly")

    def everything_query(self) -> str:
        return '*'

    @classmethod
    def _assemble_and_chunk_query_str(cls, base_query: str, chunk: bool = True, **kwargs: Any) -> list[str]:
        """
        If a query string is too long, we can attempt to run it anyway by splitting the domain substring (which is
        guaranteed to be only a sequence of ANDs) into parts, to produce multiple smaller queries which are collectively
        equivalent to the original.

        Because we have this chunking thing implemented, and the filter behavior never interacts with the domain search
        behavior, we can just put the two different search fields into two different sets of behavior at the top.
        There's obvious room to optimize, but this gets the done job.
        """
        cls.trace(Trace.QSTR, "AP._assemble_and_chunk_query_str %s %s %r", base_query, chunk, kwargs)
        domains = kwargs.get('domains', [])

        filters = kwargs.get('filters', [])

        if chunk and (len(base_query) > cls.MAX_QUERY_LENGTH):
            # of course there still is the possibility that the base query is too large, which
            # cannot be fixed by this method
            raise RuntimeError(f"Base Query cannot exceed {cls.MAX_QUERY_LENGTH} characters")

        # Get Domain Queries
        domain_queries = []
        if len(domains) > 0:
            domain_queries = [cls._assembled_query_str(base_query, domains=domains)]
            domain_queries_too_big = any([len(q_) > cls.MAX_QUERY_LENGTH for q_ in domain_queries])

            domain_divisor = 2

            if chunk and domain_queries_too_big:
                while domain_queries_too_big:
                    chunked_domains = np.array_split(domains, domain_divisor)
                    domain_queries = [cls._assembled_query_str(base_query, domains=dom) for dom in chunked_domains]
                    domain_queries_too_big = any([len(q_) > cls.MAX_QUERY_LENGTH for q_ in domain_queries])
                    domain_divisor *= 2
                
        # Then Get Filter Queries
        filter_queries = []
        if len(filters) > 0:
            filter_queries = [cls._assembled_query_str(base_query, filters=filters)]
            filter_queries_too_big = any([len(q_) > cls.MAX_QUERY_LENGTH for q_ in filter_queries])

            filter_divisor = 2
            if chunk and filter_queries_too_big:
                while filter_queries_too_big:
                    chunked_filters = np.array_split(filters, filter_divisor)
                    filter_queries = [cls._assembled_query_str(base_query, filters=filt) for filt in chunked_filters]
                    filter_queries_too_big = any([len(q_) > cls.MAX_QUERY_LENGTH for q_ in filter_queries])
                    filter_divisor *= 2
            
        # There's a (probably not uncommon) edge case where we're searching against no collections at all,
        # so just do it manually here.
        if len(domain_queries) == 0 and len(filter_queries) == 0:
            queries = [cls._assembled_query_str(base_query)]
        
        else:
            queries = domain_queries + filter_queries
        
        return queries

    @staticmethod
    def _prune_kwargs(kwargs: dict[str, Any]) -> None:
        """
        takes a query **kwargs dict and removes keys that
        are processed in this library, and should not be passed to clients.
        """
        kwargs.pop("chunk", None) # bool
        kwargs.pop("domains", None) # Iterable[str]
        kwargs.pop("filters", None) # Iterable[str]
        kwargs.pop("url_search_strings", None) # dict[str, Iterable[str]]
        kwargs.pop("url_search_string_domain", None) # bool: TEMP

    @classmethod
    def _check_kwargs(cls, kwargs: dict[str, Any]) -> None:
        """
        check for unknown/misspelled kwargs

        called with kwargs dict after query arguments removed
        copyies kwargs, removes local-only keys and raises
        exception if anything remains
        """
        kwcopy = kwargs.copy()
        cls._prune_kwargs(kwcopy)
        if kwcopy:
            exstring = ", ".join(kwcopy) # join key names
            # If here with "_seconds", client's cache_function needs updating!
            raise TypeError(f"unknown keyword args: {exstring}")

    @classmethod
    def _assemble_and_chunk_query_str_kw(cls, base_query: str, kwargs: dict = {}) -> list[str]:
        """
        takes kwargs as *dict*, removes items that shouldn't be sent to _client
        """
        chunk = kwargs.pop("chunk", True)
        queries = cls._assemble_and_chunk_query_str(base_query, chunk=chunk, **kwargs)
        cls._prune_kwargs(kwargs)
        return queries

    @classmethod
    def _selector_query_clauses(cls, kwargs: dict) -> list[str]:
        """
        take domains and filters kwargs and
        returns a list of query_strings to be OR'ed together
        (to be AND'ed with user query *or* used as a filter)
        """
        cls.trace(Trace.QSTR, "AP._selector_query_clauses IN: %r", kwargs)
        selector_clauses = []

        domains = kwargs.get('domains', [])
        if len(domains) > 0:
            domain_strings = " OR ".join(domains)
            selector_clauses.append(f"{cls.domain_search_string()}:({domain_strings})")
            
        # put all filters in single query string
        # (NOTE: filters are additive, not subtractive!)
        filters = kwargs.get('filters', [])
        if len(filters) > 0:
            for filter in filters:
                if "AND" in filter:
                    # parenthesize if any chance it has a grabby AND.
                    # (Phil: did I get this in reverse? and would need to parenthesize
                    # things containing OR if ANDing subtractive clauses together?)
                    selector_clauses.append(f"({filter})")
                else:
                    selector_clauses.append(filter)
        cls.trace(Trace.QSTR, "AP._selector_query_clauses OUT: %s", selector_clauses)
        return selector_clauses

    @classmethod
    def _selector_count(cls, kwargs: dict) -> int:
        return len(kwargs.get('domains', [])) + len(kwargs.get('filters', []))

    @classmethod
    def _selector_query_string_from_clauses(cls, clauses: list[str]) -> str:
        return " OR ".join(clauses)

    @classmethod
    def _selector_query_string(cls, kwargs: dict) -> str:
        """
        takes kwargs (as dict) return a query_string to be AND'ed with
        user query or used as a filter.
        """
        return cls._selector_query_string_from_clauses(cls._selector_query_clauses(kwargs))

    @classmethod
    def _assembled_query_str(cls, query: str, **kwargs: Any) -> str:
        cls.trace(Trace.QSTR, "_assembled_query_str IN: %s %r", query, kwargs)
        sqs = cls._selector_query_string(kwargs) # takes dict
        if sqs:
            q = f"({query}) AND ({sqs})"
        else:
            q = query
        cls.trace(Trace.QSTR, "_assembled_query_str OUT: %s", q)
        return q

    def __repr__(self) -> str:
        # important to keep this unique among platforms so that the caching works right
        return type(self).__name__


class OnlineNewsWaybackMachineProvider(OnlineNewsAbstractProvider):
    """
    All these endpoints accept a `domains: List[str]` keyword arg.
    """
    BASE_URL = ""               # SearchApiClient has default
    STAT_NAME = "wbm"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        # NOTE! typing _client as Any to avoid type checking client calls
        self._client = SearchApiClient("mediacloud", self._base_url)
        if self._timeout:
            self._client.TIMEOUT_SECS = self._timeout

    @classmethod
    def domain_search_string(cls) -> str:
        return "domain"

    # Chunk'd
    # NB: it looks like the limit keyword here doesn't ever get passed into the query - something's missing here.
    def sample(self, query: str, start_date: dt.datetime, end_date: dt.datetime, limit: int = SAMPLE_LIMIT,
               **kwargs: Any) -> Items:
        results = []
        with self._count_time("sample"):
            for subquery in self._assemble_and_chunk_query_str_kw(query, kwargs):
                this_results = self._client.sample(subquery, start_date, end_date, **kwargs)
                results.extend(this_results)

        if len(results) > limit:
            results = random.sample(results, limit)

        return self._matches_to_rows(results)

    # Chunk'd
    def count(self, query: str, start_date: dt.datetime, end_date: dt.datetime, **kwargs: Any) -> int:
        count = 0
        with self._count_time("count"):
            for subquery in self._assemble_and_chunk_query_str_kw(query, kwargs):
                count += self._client.count(subquery, start_date, end_date, **kwargs)
        return count

    # Chunk'd
    def count_over_time(self, query: str, start_date: dt.datetime, end_date: dt.datetime, **kwargs: Any) -> CountOverTime:
        counter: Counter = Counter()
        with self._count_time("count-over-time"):
            for subquery in self._assemble_and_chunk_query_str_kw(query, kwargs):
                res = self._client.count_over_time(subquery, start_date, end_date, **kwargs)
            countable = {i['date']: i['count'] for i in res}
            counter += Counter(countable)

        results = [Date(date=date, timestamp=date.timestamp(), count=count)
                   for date, count in counter.items()]
        # Sorting before returning for the sake of testability
        results.sort(key=lambda x: x["timestamp"])
        return CountOverTime(counts=results)

    @CachingManager.cache()
    def item(self, item_id: str) -> Item:
        with self._count_time("item"):
            one_item = self._client.article(item_id)
        return self._match_to_row(one_item)

    # Chunk'd
    def all_items(self, query: str, start_date: dt.datetime, end_date: dt.datetime, page_size: int = 1000, **kwargs: Any) -> AllItems:
        with self._count_time("all-items"):
            for subquery in self._assemble_and_chunk_query_str(query, **kwargs):
                for page in self._client.all_articles(subquery, start_date, end_date, **kwargs):
                    yield self._matches_to_rows(page)

    def paged_items(self, query: str, start_date: dt.datetime, end_date: dt.datetime, page_size: int = 1000, **kwargs: Any)\
            -> tuple[List[Dict], Optional[str]] :
        """
        Note - this is not chunk'd so you can't run giant queries page by page... use `all_items` instead.
        This kwargs should include `pagination_token`, which will get relayed in to the api client and fetch
        the right page of results.
        """
        updated_kwargs = {**kwargs, 'chunk': False}
        with self._count_time("paged-items"):
            query = self._assemble_and_chunk_query_str_kw(query, updated_kwargs)[0]
            page, pagination_token = self._client.paged_articles(query, start_date, end_date, **kwargs)
        return self._matches_to_rows(page), pagination_token

    # Chunk'd
    def languages(self, query: str, start_date: dt.datetime, end_date: dt.datetime, limit: int = LANGUAGES_LIMIT,
                  **kwargs: Any) -> List[Language]:

        with self._count_time("languages"):
            matching_count = self.count(query, start_date, end_date, **kwargs)

            results_counter: Counter = Counter({})
            for subquery in self._assemble_and_chunk_query_str_kw(query, kwargs):
                this_languages = self._client.top_languages(subquery, start_date, end_date, **kwargs)
                countable = {item["name"]: item["value"] for item in this_languages}
                results_counter += Counter(countable)

        # if client returns aggregated count of languages across all documents,
        # no rounding should be applied (exact counts)
        top_languages = [Language(language=name, value=value,
                                  ratio=value/matching_count, sample_size=matching_count)
                         for name, value in results_counter.items()]

        # Sort by count
        top_languages = sorted(top_languages, key=lambda x: x['value'], reverse=True)
        return top_languages[:limit]

    # Chunk'd
    def sources(self, query: str, start_date: dt.datetime, end_date: dt.datetime, limit: int = SOURCES_LIMIT,
                **kwargs: Any) -> List[Source]:

        results_counter: Counter = Counter({})
        with self._count_time("sources"):
            for subquery in self._assemble_and_chunk_query_str_kw(query, kwargs):
                results = self._client.top_sources(subquery, start_date, end_date)
                countable = {source['name']: source['value'] for source in results}
                results_counter += Counter(countable)

        cleaned_sources = [Source(source=source, count=count) for source, count in results_counter.items()]
        cleaned_sources = sorted(cleaned_sources, key=lambda x: x['count'], reverse=True)
        return cleaned_sources

    @classmethod
    def _matches_to_rows(cls, matches: List) -> Items:
        return [OnlineNewsWaybackMachineProvider._match_to_row(m) for m in matches]

    @classmethod
    def _match_to_row(cls, match: Dict) -> Dict:
        return {
            'media_name': match['domain'],
            'media_url': "http://"+match['domain'],
            'id': match['archive_playback_url'].split("/")[4],  # grabs a unique id off archive.org URL
            'title': match['title'],
            'publish_date': dateparser.parse(match['publication_date']),
            'url': match['url'],
            'language': match['language'],
            'archived_url': match['archive_playback_url'],
            'article_url': match['article_url'],
        }

    def words(self, query: str, start_date: dt.datetime, end_date: dt.datetime, limit: int = WORDS_LIMIT,
              **kwargs: Any) -> Terms:
        raise PermanentProviderException("Top Words results are not supported for Wayback Machine queries at this time")

################
# helpers for formatting url_search_strings (only enabled for MC)
# the helpers are only needed because of the TEMP url_search_string_domain

def format_and_append_uss(uss: str, url_list: list[str]) -> None:
    """
    The ONE place that knows how to format a url_search_string!!!
    (ie; what to put before and after one).

    Appends to `url_list` argument!

    NOTE! generates "unsanitized" (unsanitary?) strings!!

    Currently (11/2024) A URL Search String should:
    1. Start with fully qualified domain name WITHOUT http:// or https://
    2. End with "*"
    """
    # currently url_search_strings MUST start with fully
    # qualified domain name (FQDN) without scheme or
    # leading slashes, and MUST end with a *!
    if not uss.endswith("*"):
        uss += "*"
    url_list.append(f"http\\://{uss}")
    url_list.append(f"https\\://{uss}")

def match_formatted_search_strings(fuss: list[str]) -> str:
    """
    takes list of url search_string formatted by `format_and_append_uss`
    returns query_string fragment
    """
    assert fuss
    urls_str = " OR ".join(fuss)
    return f"url:({urls_str})"

################################################################
# here with code dragged up from mediacloud.py and news-search-api/api.py

# imports here in case split out into its own file
import base64
import json
import time
from enum import Enum
from typing import TypeAlias, cast

import elasticsearch
from elasticsearch_dsl import Search, Response
from elasticsearch_dsl.document_base import InstrumentedField
from elasticsearch_dsl.function import RandomScore
from elasticsearch_dsl.query import FunctionScore, Match, Range, Query, QueryString
from elasticsearch_dsl.response import Hit
from elasticsearch_dsl.utils import AttrDict

from .exceptions import MysteryProviderException, ProviderParseException, PermanentProviderException, TemporaryProviderException

ES_Fieldname: TypeAlias = str | InstrumentedField # quiet mypy complaints
ES_Fieldnames: TypeAlias = list[ES_Fieldname]

class FilterTuple(NamedTuple):
    weighted: int               # apply smaller values (result sets) first
    query: Query | None

_ES_MAXPAGE = 1000              # define globally (ie; in .providers)???

# Was publication_date, but web-search always passes indexed_date.
# identical indexed_date values (without fractional seconds?!)  have
# been seen in the wild (entire day 2024-01-10).  NOTE! Mapping/index
# indexed_date is now ns to reflect stored document time with μs.
_DEF_SORT_FIELD = "indexed_date"
_DEF_SORT_ORDER = "desc"

# Secondary sort key to break ties
# (see above about identical indexed_date values)
# https://www.elastic.co/guide/en/elasticsearch/reference/current/sort-search-results.html
#
# But at
# https://www.elastic.co/guide/en/elasticsearch/reference/current/paginate-search-results.html
#   "Elasticsearch uses Lucene’s internal doc IDs as tie-breakers. These
#   internal doc IDs can be completely different across replicas of the
#   same data. When paging search hits, you might occasionally see that
#   documents with the same sort values are not ordered consistently."
#
# HOWEVER: use of session_id/preference should route all requests
# from the same session to the same shards for each successive query,
# so (to quote HHGttG) "mostly harmless"?
_SECONDARY_SORT_ARGS = {"_doc": "asc"}

class SanitizedQueryString(QueryString):
    """
    query string (expression) with quoting
    """
    def __init__(self, query: str, **kwargs: Any):
        # Default allow_leading_wildcard to False.  Leading wildcards
        # kill the ES server; It's _possible_ using "wildcard" mapping
        # for url will make them usable for url_search_strings, so
        # allow override;
        if "allow_leading_wildcard" not in kwargs:
            kwargs["allow_leading_wildcard"] = False

        # quote slashes to avoid interpretation as /regexp/
        # (which not only appear in URLs but are expensive as well)
        # as done by _sanitize_es_query in mc_providers/mediacloud.py client library
        sanitized = query.replace("/", r"\/")
        super().__init__(query=sanitized, **kwargs)

class Include(Enum):
    DEFAULT = 0                 # include by default
    EXPANDED = 1                # include if expanded=True
    OPTIONAL = 2                # include if requested

# Added for format_match_fields, which was added for random_sample
# NOTE! full_language and original_url are NOT included,
# since they're never returned in a "row".
class _ES_Field:
    """ordinary field"""

    def __init__(self, field_name: str,
                 *,
                 metadata: bool = False,
                 include: Include = Include.DEFAULT):
        self.es_field_name = field_name
        self.metadata = metadata
        self.include = include

    def get(self, hit: Hit) -> Any:
        if self.metadata:
            # metadata field (incl 'id', 'index', 'score')
            return getattr(hit.meta, self.es_field_name)
        else:
            return getattr(hit, self.es_field_name)

    def convert(self, datum: Any) -> Any:
        return datum

    def get_convert(self, hit: Hit) -> Any:
        return self.convert(self.get(hit))

class _ES_DateTime(_ES_Field):
    def convert(self, datum: Any) -> Any:
        return ciso8601.parse_datetime(datum + "Z")

class _ES_Date(_ES_Field):
    def convert(self, datum: Any) -> Any:
        return dt.date.fromisoformat(datum[:10])

def _format_day_counts(bucket: list) -> Counts:
    """
    from news-search-api/client.py EsClientWrapper.format_count

    used to format "dailycounts" aggregation result

    takes [{"key_as_string": "YYYY-MM-DDT00:00:00.000Z", "doc_count": count}, ....]
    and returns {"YYYY-MM-DD": count, ....}
    """
    return {item["key_as_string"][:10]: item["doc_count"] for item in bucket}

def _format_counts(bucket: list) -> Counts:
    """
    from news-search-api/client.py EsClientWrapper.format_count

    used to format "topdomains" & "toplangs" aggregation results

    takes [{"key": key, "doc_count": doc_count}, ....]
    and returns {key: count, ....}
    """
    return {item["key"]: item["doc_count"] for item in bucket}

def _b64_encode_page_token(strng: str) -> str:
    return base64.b64encode(strng.encode(), b"-_").decode().replace("=", "~")

def _b64_decode_page_token(strng: str) -> str:
    return base64.b64decode(strng.replace("~", "=").encode(), b"-_").decode()

# Used to concatenate multiple sort keys (before b64 encoding) and split
# after b64 decode.  Must not appear in key values!  Can be multi-character
# string to lower likelihood of appearing (default keys are numeric).
_SORT_KEY_SEP = "\x01"

ES_NODE_FORMAT = "http://es{:02d}.newsscribe.angwin:9209"
ES_NODES = 8

NS_PER_SEC = 1000000000

class OnlineNewsMediaCloudProvider(OnlineNewsAbstractProvider):
    """
    version of MC Provider going direct to ES.

    Consolidates query formatting/creation previously spread
    across multiple files:

    * web-search/mcweb/backend/search/utils.py (url_search_strings)
    * this file (domain search string)
    * mc-providers/mc_providers/mediacloud.py (date ranges)
    * news-search-api/api.py (aggregation result handling)
    * news-search-api/client.py (DSL, including aggegations)

    NOTE!!! Uses elasticsearch-dsl library as much as possible (rather
    than hand-formatted JSON/dicts to allow maximum mypy type
    enforcement!!!  Passing raw JSON means ES may silently not do what
    you hoped/expected, or may cause ES runtime errors that could have
    been detected earlier.
    """

    # default values for _env_XXX calls (in alphabetical order):
    BASE_URL = ",".join(ES_NODE_FORMAT.format(n) for n in range(1,ES_NODES+1))
    INDEX_PREFIX = "mc_search"
    TIME_BY_OP = 1          # individual query timings
    USE_SUBINDEX_LIST = 0   # default to searching all ILM sub-indices

    # overrides:
    STAT_NAME = "es"
    WORDS_SAMPLE = 5000

    # elasticsearch ApiError meta.status codes to translate to TemporaryProviderException
    APIERROR_STATUS_TEMPORARY = [408, 429, 502, 503, 504]

    # map external ("row") field name to _ES_Field instance
    # (with "get" and "convert" methods to fetch/parse field from Hit)
    _ES_FIELDS: dict[str, _ES_Field] = {
        "id": _ES_Field("id", metadata=True),
        "indexed_date": _ES_DateTime("indexed_date"), # date_nanos
        "language": _ES_Field("language"),
        "media_name": _ES_Field("canonical_domain"),
        "media_url": _ES_Field("canonical_domain"),
        "publish_date": _ES_Date("publication_date"),
        "text": _ES_Field("text_content", include=Include.EXPANDED),
        "title": _ES_Field("article_title"),
        "url": _ES_Field("url"),
    }

    def __init__(self, **kwargs: Any):
        """
        Supported kwargs:

        "profile": bool or str
            if True, request profiling data, and log total ES CPU usage
            CAN pass string (filename) here, but feeding all the
            resulting JSON files to es-tools/collapse-esperf.py for
            flamegraphing could get you a mish-mash of different
            queries' results.
        "software_id": str (may be displayed by "mc-es-top" as "opaque_id")
        "session_id": str (user/session id for routing/caching)
        """

        self._profile: str | bool = kwargs.pop("profile", False)
        self._profile_current_search: str | bool = False

        # total seconds from the last profiled query:
        self._last_elastic_ms = -1.0

        # maybe take comma separated list?
        self._index = self._env_str(kwargs.pop("index_prefix", None), "INDEX_PREFIX") + "-*"

        self._use_subindex_list = self._env_int(kwargs.pop("use_subindex_list", None),
                                                "USE_SUBINDEX_LIST")

        # after pop-ing any local-only args:
        super().__init__(**kwargs)

        eshosts = self._base_url.split(",") # comma separated list of http://SERVER:PORT

        # Retries without delay (never mind backoff!)
        # web-search creates new Provider for each API request,
        # so randomize the pool.

        # https://www.elastic.co/guide/en/elasticsearch/reference/current/api-conventions.html
        # says:
        #   The X-Opaque-Id header accepts any arbitrary
        #   value. However, we recommend you limit these values to a
        #   finite set, such as an ID per client. Don’t generate a
        #   unique X-Opaque-Id header for every request. Too many
        #   unique X-Opaque-Id values can prevent Elasticsearch from
        #   deduplicating warnings in the deprecation logs.
        # See session_id for per-user/instance identification.

        self._es = elasticsearch.Elasticsearch(eshosts,
                                               max_retries=3,
                                               opaque_id=self._software_id,
                                               request_timeout=self._timeout,
                                               randomize_nodes_in_pool=True)


    @classmethod
    def domain_search_string(cls) -> str:
        return "canonical_domain"

    @classmethod
    def _selector_query_clauses(cls, kwargs: dict) -> list[str]:
        """
        take domains, filters, url_search_strings as kwargs
        return a list of query_strings to be OR'ed together
        (to be AND'ed with user query or used as a filter)
        """
        cls.trace(Trace.QSTR, "MC._selector_query_clauses IN: %r", kwargs)
        selector_clauses = super()._selector_query_clauses(kwargs)

        # Here to try to get web-search out of query
        # formatting biz.  Accepts a Mapping indexed by
        # domain_string, of lists (or sets!) of search_strings.
        url_search_strings: UrlSearchStrings = kwargs.get('url_search_strings', {})
        if url_search_strings:
            # Unclear if domain field check actually helps at all,
            # so make it optional for testing.
            if kwargs.get("url_search_string_domain", True): # TEMP: include canonincal_domain:
                domain_field = cls.domain_search_string()

                # here with mapping of cdom => iterable[search_string]
                for cdom, search_strings in url_search_strings.items():
                    fuss: List[str] = [] # formatted url_search_strings
                    for sstr in search_strings:
                        format_and_append_uss(sstr, fuss)

                    mfuss = match_formatted_search_strings(fuss)
                    selector_clauses.append(
                        f"({domain_field}:{cdom} AND {mfuss})")

                    format_and_append_uss(cdom, fuss)
            else: # make query without domain (name) field check
                # collect all the URL search strings
                fuss = []
                for cdom, search_strings in url_search_strings.items():
                    for sstr in search_strings:
                        format_and_append_uss(sstr, fuss)

                # check all the urls in one swell foop!
                selector_clauses.append(
                        match_formatted_search_strings(fuss))

        cls.trace(Trace.QSTR, "MC._selector_query_clauses OUT: %s", selector_clauses)
        return selector_clauses

    @classmethod
    def _selector_count(cls, kwargs: dict) -> int:
        url_search_strings: UrlSearchStrings = kwargs.get('url_search_strings', {})
        count = super()._selector_count(kwargs)
        if url_search_strings:
            count += sum(map(len, url_search_strings.values()))
        return count

    def count(self, query: str, start_date: dt.datetime, end_date: dt.datetime, **kwargs: Any) -> int:
        logger.debug("MC.count %s %s %s", query, start_date, end_date)
        self.trace(Trace.QSTR, "MC.count kwargs %r", kwargs)
        # no chunking on MC
        results = self._overview_query(query, start_date, end_date, **kwargs)
        return self._count_from_overview(results)

    def _count_from_overview(self, results: Overview) -> int:
        """
        used in .count() and .languages()
        """
        if self._is_no_results(results):
            logger.debug("MC._count_from_overview: no results")
            return 0
        count = results['total']
        logger.debug("MC._count_from_overview: %s", count)
        return count

    def count_over_time(self, query: str, start_date: dt.datetime, end_date: dt.datetime, **kwargs: Any) -> CountOverTime:
        logger.debug("MC.count_over_time %s %s %s", query, start_date, end_date)
        self.trace(Trace.QSTR, "MC.count_over_time kwargs %r", kwargs)

        results = self._overview_query(query, start_date, end_date, **kwargs)
        to_return: List[Date] = []
        if not self._is_no_results(results):
            data = results['dailycounts']
            # transform to list of dicts for easier use: process in sorted order
            for day_date in sorted(data):  # date is in 'YYYY-MM-DD' format
                dt = ciso8601.parse_datetime(day_date) # PB: is datetime!!
                to_return.append(Date(
                    date=dt.date(), # PB: was returning datetime!
                    timestamp=int(dt.timestamp()), # PB: conversion may be to local time!!
                    count=data[day_date]
                ))
        logger.debug("MC.count_over_time %d items", len(to_return))
        self.trace(Trace.RESULTS, "MC.count_over_time %r", to_return)
        return CountOverTime(counts=to_return)

    # using default sample & words methods (using random_sample)

    def languages(self, query: str, start_date: dt.datetime, end_date: dt.datetime, limit: int = LANGUAGES_LIMIT,
                  **kwargs: Any) -> List[Language]:
        logger.debug("MC.languages %s %s %s", query, start_date, end_date)
        self.trace(Trace.QSTR, "MC.languages kwargs %r", kwargs)
        kwargs.pop("sample_size", None)
        results = self._overview_query(query, start_date, end_date, **kwargs)
        if self._is_no_results(results):
            return []
        matches = self._count_from_overview(results)
        # NOTE! value and matches are exact (population counts, not based on sampling)
        # so they are "exact", and no rounding applied to ratio!
        top_languages = [Language(language=name, value=value, ratio=value/matches,
                                  sample_size=matches)
                         for name, value in results['toplangs'].items()]
        logger.debug("MC.languages: _overview returned %d items", len(top_languages))

        # Sort by count
        top_languages = sorted(top_languages, key=lambda x: x['value'], reverse=True)
        items = top_languages[:limit]

        logger.debug("MC.languages: returning %d items", len(items))
        self.trace(Trace.RESULTS, "MC.languages %r", items)
        return items

    def sources(self, query: str, start_date: dt.datetime, end_date: dt.datetime, limit: int = SOURCES_LIMIT,
                **kwargs: Any) -> List[Source]:
        logger.debug("MC.sources %s %s %s", query, start_date, end_date)
        self.trace(Trace.QSTR, "MC.sources kwargs %r", kwargs)
        results = self._overview_query(query, start_date, end_date, **kwargs)
        items: list[Source]
        if self._is_no_results(results):
            items = []
        else:
            cleaned_sources = [Source(source=source, count=count) for source, count in results['topdomains'].items()]
            items = sorted(cleaned_sources, key=lambda x: x['count'], reverse=True)
        logger.debug("MC.sources: %d items", len(items))
        self.trace(Trace.RESULTS, "MC.sources %r", items)
        return items

    @classmethod
    def _assemble_and_chunk_query_str(cls, base_query: str, chunk: bool = True, **kwargs: Any) -> list[str]:
        """
        Called by OnlineNewsAbstractProvider.all_items, .words;
        ignores chunking!
        """
        cls.trace(Trace.QSTR, "MC._assemble_and_chunk_query_str %s %s %r", base_query, chunk, kwargs)
        return [cls._assembled_query_str(base_query, **kwargs)]

    def _fields(self, expanded: bool) -> ES_Fieldnames:
        """
        originally in news-search-api/client.py QueryBuilder constructor:
        return list of ES fields for item, paged_items, all_items to return.

        Now using _ES_FIELDS; see also fields method (returns external names)
        """
        fields: ES_Fieldnames = [
            f.es_field_name
            for f in self._ES_FIELDS.values()
            if (f.include == Include.DEFAULT or
                (expanded and f.include == Include.EXPANDED))
        ]
        return fields

    # Multipliers to allow weighting in order to (experimentally)
    # apply filters in most efficient order (cheapest/most selective
    # filter first).  If all sources and all days were equal they
    # would be equally selective. BUT adding a day means only
    # expanding a range. _LOWER_ values mean filter applied first.
    # So test increasing SELECTOR_WEIGHT?
    SELECTOR_WEIGHT = 1         # domains, filters, url_search_strings
    DAY_WEIGHT = 1

    @classmethod
    def _selector_filter_tuple(cls, kwargs: dict) -> FilterTuple:
        """
        function to allow construction of DSL
        """

        # rather than restorting to formatting/quoting query-string
        # only to have ES have to parse it:
        # For canonical_domain: "Match" query defaults to OR for space separated words
        # For url: use "Wildcard"??
        # Should initially take (another) temp kwarg bool to allow A/B testing!!!
        # elasticsearch_dsl allows "Query | Query"

        selector_clauses = cls._selector_query_clauses(kwargs)
        if selector_clauses:
            sqs = cls._selector_query_string_from_clauses(selector_clauses)
            return FilterTuple(cls._selector_count(kwargs) * cls.SELECTOR_WEIGHT,
                               SanitizedQueryString(query=sqs,
                                                    allow_leading_wildcard=True))
        else:
            # return dummy record, will be weeded out
            return FilterTuple(0, None)

    def _basic_search(self, user_query: str, start_date: dt.datetime, end_date: dt.datetime,
                     expanded: bool = False, source: bool = True, **kwargs: Any) -> Search:
        """
        from news-search-api/api.py cs_basic_query
        create a elasticsearch_dsl query from user_query, date range, and kwargs
        """
        # works for date or datetime! publication_date is just YYYY-MM-DD
        start = start_date.strftime("%Y-%m-%d")
        end = end_date.strftime("%Y-%m-%d")

        self._profile_current_search = kwargs.pop("profile", self._profile)

        # check for extraneous arguments
        self._check_kwargs(kwargs)

        s = Search(index=self._index_from_dates(start_date, end_date), using=self._es)

        if self._profile_current_search:
            s = s.extra(profile=True)

        if user_query.strip() != self.everything_query(): # not "*"?
            s = s.query(SanitizedQueryString(query=user_query, default_field="text_content", default_operator="and"))

        if self._session_id:
            # pass user-id and/or session
            #   id to maximize ES caching effectiveness.
            # https://www.elastic.co/guide/en/elasticsearch/reference/7.17/search-search.html#search-preference
            #   If the cluster state and selected shards do not
            #   change, searches using the same <custom-string> value
            #   (that does not start with "_") are routed to the same
            #   shards in the same order.
            s = s.params(preference=self._session_id)

        # Evaluating selectors (domains/filters/url_search_strings) in "filter context";
        # Supposed to be faster, and enable caching of document set.
        # https://www.elastic.co/guide/en/elasticsearch/reference/current/query-filter-context.html#filter-context

        # Try to apply filter with the smallest result set (most selective) first,
        # to cut down document set as soon as possible.

        days = (end_date - start_date).days + 1
        filters : list[FilterTuple] = [
            FilterTuple(days * self.DAY_WEIGHT, Range(publication_date={'gte': start, "lte": end})),
            self._selector_filter_tuple(kwargs)
            # could include languages (etc) here
        ]

        # key function avoids attempts to compare Query objects when tied!
        filters.sort(key=lambda ft : ft.weighted)
        for ft in filters:
            if ft.query:
                # ends up as list under bool.filter:
                s = s.filter(ft.query)

        if source:              # return source (fields)?
            return s.source(self._fields(expanded))
        else:
            return s.source(False) # no source fields in hits

    def _is_no_results(self, results: Overview) -> bool:
        """
        used to test _overview_query results
        """
        # or len(results["hits"]) == 0
        return results["total"] == 0

    def _get_last_indexed(self, name: str) -> str:
        """
        Called from _get_subindices (caches results for all indices).
        returns max indexed_date in a (sub)index.
        only takes a few milliseconds.
        """
        AGG = "max_indexed_date"

        search = Search(index=name, using=self._es).extra(size=0) # just aggs
        search.aggs.bucket(AGG, 'max', field='indexed_date')
        with self._count_time("get-last-indexed"):
            res = search.execute()
        # XXX maybe call _check_response(res)??
        t = res.aggregations[AGG].value_as_string
        assert isinstance(t, str)
        return t[:10]           # YYYY-MM-DD

    @CachingManager.cache(seconds=15*60)
    def _get_subindices(self, index: str) -> list[list[str]]:
        """
        index is a wildcard, passed as argument (instead of
        picked up from self._index) so included in hash key
        for paranoia in case index switched for testing!!!

        returns ordered list of [last_indexed_date_str, subindex_name]
        sorted in descending order
        """
        # "indices.get" returns lots of data, none of it useful
        # (creation date could be when it was reindexed!)
        res = self._es.indices.get_alias(index=index)

        # plain list of lists, so JSONable.  date first for sorting below
        subindices = [
            [self._get_last_indexed(name), name]
            for name in res.keys()
        ]

        # sort in descending order by date of last indexed story
        # should be same as descending sort on name!!!
        subindices.sort(reverse=True)
        self.trace(Trace.SUBINDICES, "subindices %s", subindices)
        return subindices

    def _index_from_dates(self, start_date: dt.datetime, end_date: dt.datetime) -> list[str]:
        """
        return list of indices to search for a given date range.
        if indexing goes back to being split by publication_date (by year or quarter?)
        this could limit the number of shards that need to be queried
        """
        if self._use_subindex_list:
            # expand by a month for: articles accepted in advance,
            # date truncation in subindices list, subindex overlap
            start_pub_datetime = start_date - dt.timedelta(days=31)
            start_pub_date_str = start_pub_datetime.date().isoformat()

            try:
                ret: list[str] = []
                for last_indexed, name in self._get_subindices(self._index):
                    # quit as soon as next oldest index can't contain anything
                    if start_pub_date_str > last_indexed:
                        break
                    ret.append(name)
                self.trace(Trace.SUBINDICES, "subindex_list %s %r", start_pub_date_str, ret)
                return ret
            except (elasticsearch.exceptions.TransportError, elasticsearch.exceptions.ApiError):
                pass
        return [self._index]    # return list with wildcard

    def _search(self, search: Search, op: str) -> Response:
        """
        one place to send queries to ES, for logging
        """
        execute_args = {}
        if self._caching < 0:
            # Here to try to force ES not to use cached results (for testing).
            # Only effects in-library caching:
            execute_args["ignore_cache"] = True

            # This puts ?request_cache=false on the request URL, which
            # https://www.elastic.co/guide/en/elasticsearch/reference/current/shard-request-cache.html
            # says "The request_cache query-string parameter can be
            # used to enable or disable caching on a per-request
            # basis. If set, it overrides the index-level setting"
            search = search.params(request_cache=False)

        if self.trace_enabled(Trace.RAW_QUERY):
            self.trace(Trace.RAW_QUERY, "query %r", search.to_dict())

        try:
            with self._count_time(op):
                res = search.execute(**execute_args)
        except elasticsearch.exceptions.TransportError as e:
            logger.debug("%r: %r", e, search.to_dict())
            raise TemporaryProviderException("networking") from e
        except elasticsearch.exceptions.ApiError as e:
            logger.debug("%r: %r", e, search.to_dict())
            # Messages will almost certainly need massage to be
            # end-user friendly!  It would be preferable to translate
            # them here, but it will require time to acquire the
            # (arcane) knowledge and experience.
            try:
                for cause in e.body["error"]["root_cause"]:
                    short = cause["type"]
                    long = cause["reason"]
                    if short == "parse_exception":
                        raise self._parse_exception(long)
            except (LookupError, TypeError):
                logger.debug("could not get root_cause: %r", e.body)
                short = str(e)
                long = repr(e)

            if e.error in self.APIERROR_STATUS_TEMPORARY:
                raise TemporaryProviderException(short, long) from e
            raise PermanentProviderException(short, long) from e

        logger.debug("MC._search ES took %s ms", getattr(res, "took", -1))
        if self.trace_enabled(Trace.RAW_RESPONSE):
            self.trace(Trace.RAW_RESPONSE, "response %r", res.to_dict())

        if (pdata := getattr(res, "profile", None)):
            self._process_profile_data(pdata)  # displays ES total time

        # look for circuit breaker trips, etc
        self._check_response(res)

        return res

    def _search_hits(self, search: Search, op: str) -> list[Hit]:
        """
        perform search, return list of Hit
        """
        res = self._search(search, op)
        return res.hits

    def _process_profile_data(self, pdata: AttrDict) -> None:
        """
        digest profiling data
        """
        pcs = self._profile_current_search # saved by _basic_search
        if isinstance(pcs, str):  # filename prefix?
            fname = time.strftime(f"{pcs}-%Y-%m-%d-%H-%M-%S.json")
            with open(fname, "w") as f:
                json.dump(pdata.to_dict(), f)
            logger.info("wrote profiling data to %s", fname)

        # sum up ES internal times
        query_ns = rewrite_ns = coll_ns = agg_ns = 0
        for shard in pdata.shards: # AttrList
            for search in shard.searches: # AttrList
                for q in search.query:    # AttrList
                    query_ns += q.time_in_nanos
                for coll in search.collector: # list
                    coll_ns += coll.time_in_nanos
                rewrite_ns += search.rewrite_time
            # XXX sum by aggregation name?
            for agg in shard.aggregations:
                agg_ns += agg.time_in_nanos
        es_nanos = query_ns + rewrite_ns + coll_ns + agg_ns
        self._last_elastic_ms = es_nanos / 1e6 # convert ns to ms
        logger.info("ES time: %.3f ms", self._last_elastic_ms)

        # avoid floating point divisions that are likely not displayed:
        # XXX save components???
        logger.debug(" ES (ns) query: %d rewrite: %d, collectors: %d aggs: %d",
                     query_ns, rewrite_ns, coll_ns, agg_ns)

    def _check_response(self, res: Response) -> None:
        """
        check for failure; try to throw a helpful exception

        NOTE!!! Because this code is complex and brittle, and the
        actual errors don't grow on trees, this method has a test
        suite of its very own, which can be run by incanting:

        venv/bin/pip install python-dotenv pytest # only needed once
        venv/bin/pytest mc_providers/test/test_onlinenews_errors.py

        The tests don't require any access to an Elasticsearch server
        (SO YOU SHOULDN'T HAVE ANY EXCUSE NOT TO RUN THEM!)

        AND, If you add code that handles new cases, please add tests!!
        """
        # Response.success() wants
        # `._shards.total == ._shards.successful and not .timed_out`
        if res.success():
            return              # our work is done!

        # see the above comment: limited to testing fields
        # that Response.success() looks at!
        shards = res._shards
        if shards.total != shards.successful:
            # process per-shard errors
            parse_error = ''
            permanent_shard_error = None

            # hundreds of shards, so summarize...
            # (almost always circuit breakers)
            reasons: Counter[str] = Counter()
            for shard in shards.failures:
                try:
                    # NOTE! ordered carefully, with things most likely to be present first
                    reason = shard.reason
                    if getattr(reason, "durability", "") == "PERMANENT" and not permanent_shard_error:
                        permanent_shard_error = shard

                    rt = reason.type
                    if rt:
                        reasons[rt] += 1
                        # below here things may not be present!
                        if "caused_by" in reason:
                            caused_by = reason.caused_by
                            if caused_by.type == "parse_exception" and not parse_error:
                                parse_error = getattr(caused_by, "reason", "parse error")
                except AttributeError as e:
                    # safety net
                    logger.debug("_check_response shard %r exception %r", shard, e)

            # have seen parse error PLUS permanent circuit breaker error!
            if parse_error:
                if len(reasons) > 1:
                    logger.debug("parse_error with others %r", reasons)
                raise self._parse_exception(parse_error)

            # after parse error
            logger.info("MC._search %d/%d shards failed; reasons: %r", shards.failed, shards.total, reasons)

            # have seen
            # type == "circuit_breaking_exception",
            # reason == "[fielddata] Data too large, data for [Global Ordinals] ....."
            # durability == "PERMANENT"
            if permanent_shard_error:
                logger.warning("permanent error %r", permanent_shard_error.to_dict())
                pser = permanent_shard_error.reason
                raise PermanentProviderException(pser.type, pser.reason)

            if "circuit_breaking_exception" in reasons:
                raise TemporaryProviderException("Out of memory")

            logger.error("Unknown response error %r", res.to_dict())
            raise MysteryProviderException(shards.failures[0].reason.type,
                                           shards.failures[0].reason.reason)
        elif res.timed_out:
            logger.info("elasticsearch response has timed_out set")
            raise TemporaryProviderException("Timed out")

        # likely here because Response.success() has changed?!
        logger.error("Unknown response error %r", res.to_dict())
        raise MysteryProviderException("Unknown error")

    @staticmethod               # for testing
    def _parse_exception(multiline: str) -> ProviderParseException:
        """
        take (multiline) parser error message, and return ProviderParseException
        """
        lines = multiline.split("\n", 1)
        first = lines[0]
        rest = lines[1:] or ""  # handle single line
        return ProviderParseException(first, rest)

    @CachingManager.cache('overview')
    def _overview_query(self, query: str, start_date: dt.datetime, end_date: dt.datetime, **kwargs: Any) -> Overview:
        """
        from news-search-api/api.py
        """

        logger.debug("MC._overview %s %s %s", query, start_date, end_date)
        self.trace(Trace.QSTR, "MC._overview kwargs %r", kwargs)

        # these are arbitrary, but match news-search-api/client.py
        # so that es-tools/mc-es-top.py can recognize this is an overview query:
        AGG_DAILY = "dailycounts"
        AGG_LANG = "toplangs"
        AGG_DOMAIN = "topdomains"

        search = self._basic_search(query, start_date, end_date, **kwargs)
        search.aggs.bucket(AGG_DAILY, "date_histogram", field="publication_date",
                           calendar_interval="day", min_doc_count=1)
        search.aggs.bucket(AGG_LANG, "terms", field="language", size=100)
        search.aggs.bucket(AGG_DOMAIN, "terms", field="canonical_domain", size=100)
        search = search.extra(track_total_hits=True, size=0)
        res = self._search(search, "overview") # run search, need .aggregations & .hits

        hits = res.hits            # property
        aggs = res.aggregations
        return Overview(
            query=query,
            # res.hits.total.value documented at
            # https://elasticsearch-dsl.readthedocs.io/en/stable/search_dsl.html#response
            total=hits.total.value, # type: ignore[attr-defined]
            topdomains=_format_counts(aggs[AGG_DOMAIN]["buckets"]),
            toplangs=_format_counts(aggs[AGG_LANG]["buckets"]),
            dailycounts=_format_day_counts(aggs[AGG_DAILY]["buckets"])
        )

    @CachingManager.cache()
    def item(self, item_id: str) -> Item:
        expanded = True         # always includes full_text!!
        s = Search(index=self._index, using=self._es)\
            .query(Match(_id=item_id))\
            .source(includes=self._fields(expanded)) 
        hits = self._search_hits(s, "item")
        if not hits:
            return {}

        return self._hit_to_row(hits[0], self.fields(expanded), True)

    def paged_items(
            self, query: str,
            start_date: dt.datetime, end_date: dt.datetime,
            page_size: int = 1000,
            **kwargs: Any
    ) -> tuple[list[dict], Optional[str]]:
        """
        return a single page of data (with `page_size` items).
        Pass `None` as first `pagination_token`, after that pass
        value returned by previous call, until `None` returned.

        `kwargs` may contain: `sort_field` (str), `sort_order` (str), `expanded` (bool)
        """
        logger.debug("MC._paged_items q: %s: %s e: %s ps: %d",
                     query, start_date, end_date, page_size)
        self.trace(Trace.QSTR, "MC._paged_items kw: %r", kwargs)

        page_size = min(page_size, _ES_MAXPAGE)
        expanded = kwargs.pop("expanded", False)
        sort_field = kwargs.pop("sort_field", _DEF_SORT_FIELD)
        sort_order = kwargs.pop("sort_order", _DEF_SORT_ORDER)
        pagination_token = kwargs.pop("pagination_token", None)

        # NOTE! depends on client limiting to reasonable choices!!
        # (full text might leak data, or causes memory exhaustion!)
        # originally took internal sort_field names only, now accept
        # both, prefering to intrepret as external first
        # (the default indexed_date name is same inside and out)
        sf = self._ES_FIELDS.get(sort_field)
        if sf and not sf.metadata:
            sort_field = sf.es_field_name
        if sort_field not in self._fields(expanded):
            raise ValueError(sort_field)
        if sort_order not in ["asc", "desc"]:
            raise ValueError(sort_order)

        # see discussion above at _SECONDARY_SORT_ARGS declaration
        sort_opts = [
            {sort_field: sort_order},
            _SECONDARY_SORT_ARGS
        ]

        search = self._basic_search(query, start_date, end_date, expanded=expanded, **kwargs)\
                     .extra(size=page_size)\
                     .sort(*sort_opts)

        if pagination_token:
            # may return multiple keys:
            after = _b64_decode_page_token(pagination_token).split(_SORT_KEY_SEP)

            # important to use `search_after` instead of 'from' for
            # memory reasons related to paging through more than 10k
            # results.
            search = search.extra(search_after=after)

        hits = self._search_hits(search, "paged-items")
        if not hits:
            return ([], None)

        new_pt: str | None = None
        if len(hits) == page_size:
            # generate paging token from all sort keys of last item:
            sort_key_vals = hits[-1].meta.sort
            # indexed_date is nanoseconds, returned as int, but
            # epoch_nanos not accepted by date parser, so format:
            if sort_field == "indexed_date":
                epoch_nanos = sort_key_vals[0]
                epoch_secs = epoch_nanos // NS_PER_SEC
                last_date = time.strftime("%Y-%m-%dT%H:%M:%S",
                                          time.gmtime(epoch_secs))
                last_nanos = epoch_nanos % NS_PER_SEC
                sort_key_vals[0] = f"{last_date}.{last_nanos:09d}Z"
            new_pt = _b64_encode_page_token(
                _SORT_KEY_SEP.join([str(key) for key in sort_key_vals]))

        fields = self.fields(expanded)
        rows = [self._hit_to_row(h, fields, True) for h in hits]
        self.trace(Trace.RESULTS, "MC next %s rows %r", new_pt, rows)
        return (rows, new_pt)

    def all_items(self, query: str,
                  start_date: dt.datetime, end_date: dt.datetime,
                  page_size: int = _ES_MAXPAGE, **kwargs: Any) -> AllItems:
        """
        returns generator of pages (lists) of items
        """
        next_page_token: str | None = None
        while True:
            page, next_page_token = self.paged_items(
                query, start_date, end_date,
                page_size=page_size,
                pagination_token=next_page_token,
                **kwargs)

            if not page:
                break

            yield page

            if not next_page_token:
                break

    @classmethod
    def _hit_to_row(cls, hit: Hit, fields: list[str], return_none: bool = False) -> dict[str, Any]:
        """
        format a Hit returned by ES into an external "row" suitable for return.
        fields is a list of external/row field names to be returned.
        if return_none is set, return None for missing results
        """
        # iterates over _external_ names rather than just returned
        # fields to be able to return metadata fields.
        res: dict[str, Any] = {}
        for field in fields:
            try:
                res[field] = cls._ES_FIELDS[field].get_convert(hit)
            except AttributeError:
                if return_none:
                    res[field] = None
        return res

    def random_sample(self, query: str, start_date: dt.datetime, end_date: dt.datetime,
                      page_size: int, fields: list[str], **kwargs: Any) -> AllItems:
        """
        Returns generator to allow pagination, but currently returns only
        single page; actual pagination may require more work (passing
        "seed" and "field" arguments to RandomSample).  Maybe foist
        issue on caller and require "seed" argument??

        To implement pagination in web-search API, would need to have
        a method here that takes and returns a pagination token that
        this method calls...  Perhaps paged_items could do the job
        when passed a "randomize" argument???
        """
        if not fields:
            raise ValueError("random_sample requires fields list")

        # max controlled by index-level index.max_result_window, default is 10K.
        # allows 5K samples for lang/title pairs (for top words):
        max_page = 10000 // len(fields)
        if page_size > max_page:
            page_size = max_page

        # convert requested external field names to ES field names to request
        es_fields: ES_Fieldnames = [
            self._ES_FIELDS[f].es_field_name
            for f in fields
            if not self._ES_FIELDS[f].metadata # no need to ask for metadata
        ]

        search = self._basic_search(query, start_date, end_date, **kwargs)\
                     .query(
                         FunctionScore(
                             functions=[
                                 RandomScore(
                                     # needed for 100% reproducibility (ie; if paging results)
                                     # seed=int, field="fieldname"
                                 )
                             ]
                         )
                     )\
                     .source(es_fields)\
                     .extra(size=page_size)

        hits = self._search_hits(search, "random-sample")
        yield [self._hit_to_row(hit, fields) for hit in hits] # just one page

    @classmethod
    def fields(cls, expanded: bool = False) -> list[str]:
        """
        returns external field names (helper for random_sample).
        see also _fields (returns internal names)
        """
        return [
            ext_name
            for ext_name, f in cls._ES_FIELDS.items()
            if (f.include == Include.DEFAULT or
                (expanded and f.include == Include.EXPANDED))
        ]

    #@CachingManager.cache() # enable if used for user-facing functions!!
    def sources_by_date(self, query: str,
                        start_date: dt.datetime, end_date: dt.datetime,
                        max_domains: Optional[int] = None,
                        interval: Optional[str] = None,
                        date_extras: dict[str, Any] = {},
                        **kwargs: Any) -> SourcesByDate:
        """
        created for Media Cloud internal directory management!
        Returns dictionary indexed by date of dicts indexed by domain of counts
        """
        AGG_DATE = 'date'
        AGG_SRCS = 'srcs'

        date_delta = end_date - start_date

        # if no interval supplied, pick one
        # (maybe truncate date range to bucket boundaries?)
        if interval is None:
            if date_delta.days <= 31:
                interval = "day"
            elif date_delta.days <= 2*365+1:
                interval = "month"
            else:
                interval = "year"

        # num_date_buckets the number of date buckets based on date range
        # by default (cluster setting), max total buckets is 65536.
        # NOTE!!! Assumes start date is bucket aligned
        # (first of month, start day of week, start of year)
        num_date_buckets: int
        if interval == "day":
            date_format = "%Y-%m-%d"
            date_delta.days
        if interval == "week":
            # supplied start/end dates should be first and last days of weeks!
            # NOTE! By default start buckets w/ monday dates.
            # Supply date_extras={"offset": "-1d"} for Sunday.
            # Partial weeks before or after will create extra buckets!!
            date_format = "%Y-%m-%d"
            num_dates = int(date_delta.days / 7 + 0.9) # WRONG!
        elif interval == "month":
            # supplied start/end dates should be first and last days of months!
            date_format = "%Y-%m"
            num_dates = ((end_date.year - start_date.year) * 12 +
                         (end_date.month - start_date.month + 1))
        elif interval == "year":
            # supplied start/end dates should be first and last days of year!!
            date_format = "%Y"
            num_dates = end_date.year - start_date.year + 1
        else:
            raise ValueError(f"unknown interval {interval}")

        if max_domains is None:
            # use reduced total bucket count to allow for
            # non-bucket-aligned dates creating extra buckets?
            max_domains = 60000 // num_dates

        search = self._basic_search(query, start_date, end_date, **kwargs)\
                     .extra(size=0) # just aggs
        # nested buckets!
        search.aggs.bucket(AGG_DATE, "date_histogram",
                           field="publication_date",
                           calendar_interval=interval,
                           **date_extras)\
                   .bucket(AGG_SRCS, "terms",
                           field="canonical_domain",
                           size=max_domains)
        res = self._search(search, "src_by_date")
        ret: SourcesByDate  = {}
        date_buckets = cast(list[dict[str, Any]], res.aggregations[AGG_DATE])
        for date in date_buckets:
            # key is epoch UTC milliseconds
            tm = time.gmtime(int(date['key']) // 1000)
            fdate = time.strftime(date_format, tm) # formatted date
            ret[fdate] = {str(b["key"]): int(b["doc_count"])
                          for b in date[AGG_SRCS]['buckets']}
        # maybe return top level TypedDict with totals (if any) returned with query result?
        return ret

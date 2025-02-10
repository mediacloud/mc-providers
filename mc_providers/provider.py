import collections
import datetime as dt
import importlib.metadata       # to get version for SOFTWARE_ID
import logging
import os
import re
from abc import ABC
from typing import Any, Generator, Iterable, NoReturn, TypeAlias, TypedDict
from operator import itemgetter

from .exceptions import MissingRequiredValue, QueryingEverythingUnsupportedQuery
from .language import terms_without_stopwords

logger = logging.getLogger(__name__) # for trace
logger.setLevel(logging.DEBUG)

# helpful for turning any date into the standard Media Cloud date format
MC_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

DEFAULT_TIMEOUT = 60  # to be used across all the providers; override via one-time call to set_default_timeout

# from api-client/.../api.py
try:
    VERSION = "v" + importlib.metadata.version("mc-providers")
except importlib.metadata.PackageNotFoundError:
    VERSION = "dev"


def set_default_timeout(timeout: int) -> None:
    global DEFAULT_TIMEOUT
    DEFAULT_TIMEOUT = timeout

Item: TypeAlias = dict[str, Any]           # differs between providers?
Items: TypeAlias = list[Item]              # page of items
AllItems: TypeAlias = Iterable[Items]      # iterable of pages

class Date(TypedDict):
    """
    element of counts list in count_over_time return
    """
    date: dt.date
    timestamp: int
    count: int

class CountOverTime(TypedDict):
    """
    return type for count_over_time
    """
    counts: list[Date]

class Language(TypedDict):
    """
    list element in return value for languages method
    """
    language: str
    value: int
    ratio: float                # really fraction?!

class Source(TypedDict):
    """
    list element in return value for sources method
    """
    source: str
    count: int

class Term(TypedDict):
    """
    list element in return value for words method
    """
    term: str
    count: int
    ratio: float                # really fraction?!

class Trace:
    # less noisy things, with lower numbers
    ARGS = 10            # constructor args
    RESULTS = 20
    QSTR = 50            # query string/args
    # even more noisy things, with higher numbers
    ALL = 1000

class ContentProvider(ABC):
    """
    An abstract wrapper to be implemented for each platform we want to preview content from.
    Any unimplemented methods raise an Exception
    """
    WORDS_SAMPLE = 500

    LANGUAGE_SAMPLE = 1000

    # default values for _env_val
    # classes which DON'T require a value should define:
    #API_KEY = ""
    #BASE_URL = ""               # subclass can override
    CACHING = 1
    SESSION_ID = ""
    SOFTWARE_ID = f"mc-providers {VERSION}"
    # TIMEOUT *NOT* defined, uses global DEFAULT_TIMEOUT below
    TRACE = 0

    _trace = int(os.environ.get("MC_PROVIDERS_TRACE", 0)) # class variable!

    def __init__(self,
                 api_key: str | None = None,
                 base_url: str | None = None,
                 timeout: int | None = None,
                 caching: int | None = None, # handles bool!
                 session_id: str | None = None,
                 software_id: str | None = None):
        """
        api_key and base_url only required by some providers, but accept for all.
        not all providers may use all values, but always accepted to be able
        to detect erroneous args!
        """
        self._api_key = self._env_str(api_key, "API_KEY")
        self._base_url = self._env_str(base_url, "BASE_URL")

        # DEFAULT_TIMEOUT possibly set using set_default_timeout
        self._timeout = self._env_int(timeout, "TIMEOUT", DEFAULT_TIMEOUT)

        if caching == 0:      # not just any falsey value
            # False ends up here, to paraphrase Mitt Romney:
            # "Yes, my friend, bools are ints"
            self._caching = 0
        else:
            self._caching = self._env_int(caching, "CACHING")

        # identify user/session making request (for caching)
        self._session_id = self._env_str(session_id, "SESSION_ID")

        # identify software making request
        # (could be used in User-Agent strings)
        self._software_id = self._env_str(software_id, "SOFTWARE_ID")

    def everything_query(self) -> str:
        raise QueryingEverythingUnsupportedQuery()

    # historically not a random semaple, see random_sample below!
    def sample(self, query: str, start_date: dt.datetime, end_date: dt.datetime, limit: int = 20,
               **kwargs: Any) -> list[dict]:
        raise NotImplementedError("Doesn't support sample content.")

    def count(self, query: str, start_date: dt.datetime, end_date: dt.datetime, **kwargs: Any) -> int:
        raise NotImplementedError("Doesn't support total count.")

    def count_over_time(self, query: str, start_date: dt.datetime, end_date: dt.datetime, **kwargs: Any) -> CountOverTime:
        raise NotImplementedError("Doesn't support counts over time.")

    def item(self, item_id: str) -> dict:
        raise NotImplementedError("Doesn't support fetching individual content.")

    def words(self, query: str, start_date: dt.datetime, end_date: dt.datetime, limit: int = 100,
              **kwargs: Any) -> list[Term]:
        raise NotImplementedError("Doesn't support top words.")

    def languages(self, query: str, start_date: dt.datetime, end_date: dt.datetime, **kwargs: Any) -> list[Language]:
        raise NotImplementedError("Doesn't support top languages.")

    def sources(self, query: str, start_date: dt.datetime, end_date: dt.datetime, limit: int = 100,
                **kwargs: Any) -> list[Source]:
        raise NotImplementedError("Doesn't support top sources.")

    def all_items(self, query: str, start_date: dt.datetime, end_date: dt.datetime, page_size: int = 1000,
                  **kwargs: Any) -> AllItems:

        # yields a page of items
        raise NotImplementedError("Doesn't support fetching all matching content.")

    def paged_items(self, query: str, start_date: dt.datetime, end_date: dt.datetime, page_size: int = 1000,
                    **kwargs: Any) -> tuple[list[dict], str | None]:
        # return just one page of items and a pagination token to get next page; implementing subclasses
        # should read in token, offset, or whatever else they need from `kwargs` to determine which page to return
        raise NotImplementedError("Doesn't support fetching all matching content.")

    def random_sample(self, query: str, start_date: dt.datetime, end_date: dt.datetime,
                      page_size: int, fields: list[str], **kwargs: Any) -> AllItems:
        # NOTE! could be subsumed by passing keyword arguments (fields, randomize) to all_items?!
        raise NotImplementedError("Doesn't support fetching random sample.")

    def normalized_count_over_time(self, query: str, start_date: dt.datetime, end_date: dt.datetime,
                                   **kwargs: Any) -> dict:
        """
        Useful for rendering attention-over-time charts with extra information suitable for normalizing
        HACK: calling _sum_count_by_date for now to solve a problem specific to the Media Cloud provider
        :param query:
        :param start_date:
        :param end_date:
        :param kwargs:
        :return:
        """
        matching_content_counts = self._sum_count_by_date(
            self.count_over_time(query, start_date, end_date, **kwargs)['counts'])
        matching_total = sum([d['count'] for d in matching_content_counts])
        no_query_content_counts = self._sum_count_by_date(
            self.count_over_time(self._everything_query(), start_date, end_date,**kwargs)['counts'])
        no_query_total = sum([d['count'] for d in no_query_content_counts])
        return {
            'counts': _combined_split_and_normalized_counts(matching_content_counts, no_query_content_counts),
            'total': matching_total,
            'normalized_total': no_query_total,
        }

    @classmethod
    def _sum_count_by_date(cls, counts: list[dict]) -> list[Date]:
        """
        Given a list of counts, sum the counts by date
        :param counts:
        :return:
        """
        counts_by_date = collections.defaultdict(int)
        for c in counts:
            date = c['date']
            counts_by_date[date] = 0
            for d in counts:
                if d['date'] == date:
                    counts_by_date[date] += d['count']
        return [{'date': d, 'count': c} for d, c in counts_by_date.items()]

    def _everything_query(self) -> str:
        """
        :return: a query string that can be used to capture matching "everything" 
        """
        return '*'

    # use this if you need to sample some content for top languages
    def _sampled_languages(self, query: str, start_date: dt.datetime, end_date: dt.datetime, limit: int = 10,
                               **kwargs: Any) -> list[Language]:
        # support sample_size kwarg
        sample_size = kwargs['sample_size'] if 'sample_size' in kwargs else self.LANGUAGE_SAMPLE
        # grab a sample and count terms as we page through it
        sampled_count = 0
        counts: collections.Counter = collections.Counter()
        for page in self.all_items(query, start_date, end_date, limit=sample_size):
            sampled_count += len(page)
            [counts.update(t.get('language', "UNK") for t in page)]
        # clean up results
        results = [Language(language=w, value=c, ratio=c/sampled_count) for w, c in counts.most_common(limit)]
        return results

    def _sample_titles(self, query: str, start_date: dt.datetime, end_date: dt.datetime, sample_size: int,
                             **kwargs: Any) -> Iterable[list[dict[str,str]]]:
        """
        default helper for _sampled_title_words: return a sampling of stories for top words
        """
        return self.random_sample(query, start_date, end_date, sample_size,
                                  fields=["title", "language"], **kwargs)

    # use this if you need to sample some content for top words
    def _sampled_title_words(self, query: str, start_date: dt.datetime, end_date: dt.datetime, limit: int = 100,
                             **kwargs: Any) -> list[Term]:
        # support sample_size kwarg
        sample_size = kwargs.pop('sample_size', self.WORDS_SAMPLE)
        # NOTE! english stopwords contain contractions!!!
        remove_punctuation = bool(kwargs.pop("remove_punctuation", True)) # XXX TEMP?

        # grab a sample and count terms as we page through it
        sampled_count = 0
        counts: collections.Counter = collections.Counter()
        for page in self._sample_titles(query, start_date, end_date, sample_size, **kwargs):
            sampled_count += len(page)
            [counts.update(terms_without_stopwords(t.get('language', 'UNK'), t['title'], remove_punctuation)) for t in page  if 'title' in t]
            needed = sample_size - sampled_count
            if needed <= 0:
                break           # quit once sample_size filled
            if needed < sample_size:
                sample_size = needed # don't overfill

        # clean up results
        results = [Term(term=w, count=c, ratio=c/sampled_count) for w, c in counts.most_common(limit)]
        self.trace(Trace.RESULTS, "_sampled_title_words %r", results)
        return results

    # from story-indexer/indexer/story.py:
    # https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
    @classmethod
    def _env_var(cls, suffix: str) -> str:
        """
        create CLASS_NAME_SUFFIX name for environment variable from class name and suffix
        """
        name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", cls.__name__)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).upper() + "_" + suffix

    def _missing_value(self, env_var: str) -> NoReturn:
        """
        stand-alone to avoid two copies of the below:

        If PROVIDER_XXX and SOURCE_XXX wanted in exception,
        define PROVIDER_NAME and SOURCE_NAME as class
        members, *BUT* likely means the
        {PROVIDER,SOURCE}_XXX variables in __init__.py would
        need to be defined in this file to avoid a dependency
        loop!  The the _NAME vars existed, they could be
        used to create the PROVIDER_MAP from just a list of
        classes, or a decorator on each class definition!
        """
        raise MissingRequiredValue(type(self).__name__, env_var)

    def _env_str(self, kwval: str | None, variable: str, default: str | None = None) -> str:
        """
        Here when no kwarg passed to constructor.
        variable should be UPPER_SNAKE_CASE string matching the kwarg name!

        NOTE! logic duplicated below in _env_int, so likely that any changes
        neeeded here should be replicated there?!

        MAYBE: prefer env (vs. kwarg likely to have been hard-coded?)
        """
        # 0. if kwarg passed, use it
        if kwval is not None: 
            self.trace(Trace.ARGS, "%r kwval %s '%s'", self, variable, kwval)
            return kwval

        env_var = self._env_var(variable)
        try:
            # 1. Look for OBJ_NAME_VARIABLE env var, returns value if it exits.
            val = os.environ[env_var]
            self.trace(Trace.ARGS, "%r env %s '%s'", self, env_var, val)
            return val
        except KeyError:
            pass

        # 2. If (run-time) default value argument passed, return it.
        if default is not None:
            self.trace(Trace.ARGS, "%r default %s '%s'", self, variable, default)
            return default

        try:
            # 3. Look for class member named "variable", if it exists, return value
            val = getattr(self, variable)
            self.trace(Trace.ARGS, "%r class default %s '%s'", self, variable, val)
            return val
        except AttributeError:
            pass

        # so not "During handling of the above exception"
        self._missing_value(env_var)

    def _env_int(self, kwval: int | None, variable: str, default: int | None = None) -> int:
        """
        ugh: copy of _env_str
        """
        if kwval is not None:
            self.trace(Trace.ARGS, "%r kwval %s %d", self, variable, kwval)
            return kwval

        env_var = self._env_var(variable)
        try:
            # 1. Look for OBJ_NAME_VARIABLE env var, returns value if it exits.
            val = int(os.environ[env_var])
            self.trace(Trace.ARGS, "%r env %s %d", self, env_var, val)
            return val
        except KeyError:
            pass

        # 2. If (run-time) default value argument passed, return it.
        if default is not None:
            self.trace(Trace.ARGS, "%r default %s %d", self, variable, default)
            return default

        try:
            # 3. Look for class member named "variable", if it exists, return value
            val = getattr(self, variable)
            self.trace(Trace.ARGS, "%r class default %s %d", self, variable, val)
            return val
        except AttributeError:
            pass

        # so not "During handling of the above exception"
        self._missing_value(env_var)

    @classmethod
    def set_trace(cls, level: int) -> None:
        cls._trace = level

    @classmethod
    def trace(cls, level: int, format: str, *args: Any) -> None:
        """
        like logger.debug, but with additional gatekeeping.  trace level
        is a class member to allow use from class methods!
        **ALWAYS** pass %-format strings plus args, to avoid formatting
        strings that are never displayed!

        See initialization of _trace above to see where the default
        value comes from
        """
        if cls._trace >= level:
            logger.debug(format, *args)

# not used??
def add_missing_dates_to_split_story_counts(counts, start, end, period="day"):
    if start is None and end is None:
        return counts
    new_counts = []
    current = start.date()
    while current <= end.date():
        date_string = current.strftime("%Y-%m-%d %H:%M:%S")
        existing_count = next((r for r in counts if r['date'] == date_string), None)
        if existing_count:
            new_counts.append(existing_count)
        else:
            new_counts.append({'date': date_string, 'count': 0})
        if period == "day":
            current += dt.timedelta(days=1)
        elif period == "month":
            current += dt.timedelta(days=31)
        elif period == "year":
            current += dt.timedelta(days=365)
        else:
            # PB: RuntimeError is for internal Python errors???
            raise RuntimeError("Unsupport time period for filling in missing dates - {}".format(period))
    return new_counts


# used in normalized_count_over_time
def _combined_split_and_normalized_counts(matching_results, total_results):
    counts = []
    for day in total_results:
        day_info = {
            'date': day['date'],
            'total_count': day['count']
        }
        matching = [d for d in matching_results if d['date'] == day['date']]
        if len(matching) == 0:
            day_info['count'] = 0
        else:
            day_info['count'] = matching[0]['count']
        if day_info['count'] == 0 or day['count'] == 0:
            day_info['ratio'] = 0
        else:
            day_info['ratio'] = float(day_info['count']) / float(day['count'])
        counts.append(day_info)
    counts = sorted(counts, key=itemgetter('date'))
    return counts

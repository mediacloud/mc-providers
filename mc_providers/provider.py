import collections
import datetime as dt
import importlib.metadata       # to get version for SOFTWARE_ID
import logging
import os
import re
import warnings
from abc import ABC
from typing import Any, Iterable, NoReturn, TypeAlias, TypedDict
from operator import itemgetter

# PyPI:
from sigfig import sigfig

from .exceptions import MissingRequiredValue, QueryingEverythingUnsupportedQuery
from .language import terms_without_stopwords_list

logger = logging.getLogger(__name__) # for trace
logger.setLevel(logging.DEBUG)

# helpful for turning any date into the standard Media Cloud date format
MC_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

DEFAULT_TIMEOUT = 60  # to be used across all the providers; override via one-time call to set_default_timeout

# default values for Provider method limit default values!
# can't be class vars, because needed in method declarations
LANGUAGES_LIMIT = 10
# used to be 20, but in practice ES returned 10
# so ... lowered expectations to avoid surprises:
SAMPLE_LIMIT = 10
SOURCES_LIMIT = 100
WORDS_LIMIT = 100

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

class _DateCount(TypedDict):
    """
    for _sum_count_by_date result
    """
    date: dt.date
    count: int

class Date(TypedDict):
    """
    element of counts list in count_over_time return
    """
    date: dt.date
    timestamp: int
    count: int

class CountOverTime(TypedDict):
    """
    return type for count_over_time method
    """
    counts: list[Date]

class _CombinedDateInfo(TypedDict, total=False):
    """
    element of counts list in normalized_count_over_time return
    returned by _combined_split_and_normalized_counts
    NOTE! total=False to allow incremental creation in existing code
    """
    date: dt.date
    total_count: int
    count: int
    ratio: float

class NormalizedCountOverTime(TypedDict):
    """
    return type for normalized_count_over_time method
    """
    counts: list[_CombinedDateInfo]
    total: int
    normalized_total: int

class Language(TypedDict):
    """
    list element in return value for languages method
    """
    language: str
    value: int
    ratio: float                # rounded
    sample_size: int

class Source(TypedDict):
    """
    list element in return value for sources method
    """
    source: str
    count: int

class _Term(TypedDict):
    """
    list element in return value for words method.
    use make_term, terms_from_counts to create
    """
    term: str
    term_count: int                  # total number of appearances
    term_ratio: float                # now rounded!
    doc_count: int              # number of documents appeared in
    doc_ratio: float            # rounded
    sample_size: int            # number of documents sampled

Terms: TypeAlias = list[_Term]

class Trace:
    # less noisy things, with lower numbers
    STATS = 5
    SUBINDICES = 8
    ARGS = 10            # constructor args
    RESULTS = 20
    QSTR = 50            # query string/args
    # even more noisy things, with higher numbers
    ALL = 1000

# disable warnings: carps about 500/1000 having one sigfig!
# https://github.com/Bobzoon/SigFigs/ implements division with sigfigs,
# but is not a PyPI package.
warnings.filterwarnings('ignore', category=UserWarning,
                        message=r".*\d+ significant figures requested from number with only \d+ .*")

def ratio_with_sigfigs(count: int, sample_size: int) -> float:
    """
    Call ONLY when using random sampling!
    try to prevent fractions with 17 or 18 digits
    (CSV and JSON formatting don't appear to do ANY rounding)
    """
    sf = min(len(str(count)), len(str(sample_size)))
    # in theory the above is correct, but force three digits:
    sf = max(sf, 3)
    return sigfig.round(count / sample_size, sigfigs=sf)

def make_term(term: str, term_count: int, doc_count: int, sample_size: int) -> _Term:
    """
    the one place to format a dict for return from "words" method
    """
    return _Term(term=term, term_count=term_count,
                 term_ratio=ratio_with_sigfigs(term_count, sample_size),
                 doc_count=doc_count,
                 doc_ratio=ratio_with_sigfigs(doc_count, sample_size),
                 sample_size=sample_size)

def terms_from_counts(term_counts: collections.Counter[str],
                      doc_counts: collections.Counter[str],
                      sample_size: int,
                      limit: int) -> Terms:
    """
    format term and doc counts for return
    """
    return [make_term(term, term_count, doc_counts[term], sample_size)
            for term, term_count in term_counts.most_common(limit)]

class ContentProvider(ABC):
    """
    An abstract wrapper to be implemented for each platform we want to preview content from.
    Any unimplemented methods raise an Exception
    """
    WORDS_SAMPLE = 500

    LANGUAGE_SAMPLE = 1000

    STAT_NAME = "FIXME"         # MUST OVERRIDE!

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

        # set in web-search config
        statsd_host = os.environ.get("STATSD_HOST")
        statsd_prefix = os.environ.get("STATSD_PREFIX")
        self._statsd_client: "statsd.StatsdClient" | None = None
        if statsd_host and statsd_prefix:
            import statsd       # avoid warnings about unclosed socket
            self._statsd_client = statsd.StatsdClient(
                statsd_host, None,
                f"{statsd_prefix}.provider.{self.STAT_NAME}")

    def everything_query(self) -> str:
        raise QueryingEverythingUnsupportedQuery()

    def _collect_random_sample(self, query: str, start_date: dt.datetime, end_date: dt.datetime,
                               samples: int, fields: list[str], **kwargs: Any) -> list[dict]:
        """
        collect `samples` random items with `fields` using random_sample.
        MAYBE enforce max samples based on number of fields 10000//len(fields)?
        """
        results: list[dict] = []
        limit = samples
        for page in self.random_sample(query, start_date, end_date, limit, fields, **kwargs):
            results.extend(page)
            limit = samples - len(results)
            if limit <= 0:
                break
        return results

    def sample(self, query: str, start_date: dt.datetime, end_date: dt.datetime, limit: int = SAMPLE_LIMIT,
               **kwargs: Any) -> list[dict]:
        return self._collect_random_sample(query, start_date, end_date, samples=limit,
                                           fields=self.fields(), **kwargs)

    def count_over_time(self, query: str, start_date: dt.datetime, end_date: dt.datetime, **kwargs: Any) -> CountOverTime:
        raise NotImplementedError("Doesn't support counts over time.")

    def item(self, item_id: str) -> Item:
        raise NotImplementedError("Doesn't support fetching individual content.")

    def sources(self, query: str, start_date: dt.datetime, end_date: dt.datetime, limit: int = SOURCES_LIMIT,
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
        # NOTE! same type signature as all_items (plus fields)
        # Could be subsumed by passing keyword arguments (fields, randomize) to all_items?!
        # A paged_ version would be needed for to expose a web-search API call
        # (which would require a "seed" value to generate a consistent sequence)
        raise NotImplementedError("Doesn't support fetching random sample.")

    @classmethod
    def fields(cls, expanded: bool = False) -> list[str]:
        """
        helper for random_sample; return list of "normal" fields
        """
        raise NotImplementedError("Doesn't support field nemaes.")

    def normalized_count_over_time(self, query: str, start_date: dt.datetime, end_date: dt.datetime,
                                   **kwargs: Any) -> NormalizedCountOverTime:
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
        return NormalizedCountOverTime(
            counts=_combined_split_and_normalized_counts(matching_content_counts, no_query_content_counts),
            total=matching_total,
            normalized_total=no_query_total,
        )

    @classmethod
    def _sum_count_by_date(cls, counts: list[Date]) -> list[_DateCount]:
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

    def languages(self, query: str, start_date: dt.datetime, end_date: dt.datetime, limit: int = LANGUAGES_LIMIT,
                               **kwargs: Any) -> list[Language]:
        """
        generic version of languages, using _collect_random_sample
        """
        # support sample_size kwarg
        sample_size = kwargs.get('sample_size', self.LANGUAGE_SAMPLE)

        # grab a sample
        sample = self._collect_random_sample(query, start_date, end_date,
                                             samples=sample_size,
                                             fields=["language"], **kwargs)
        sampled_count = len(sample)

        # get counts
        counts: collections.Counter = collections.Counter()
        counts.update(s.get('language', "UNK") for s in sample)

        # clean up results
        results = [Language(language=w,
                            value=c,
                            ratio=ratio_with_sigfigs(c, sampled_count),
                            sample_size=sampled_count)
                   for w, c in counts.most_common(limit)]
        return results

    def words(self, query: str, start_date: dt.datetime, end_date: dt.datetime, limit: int = WORDS_LIMIT,
              **kwargs: Any) -> Terms:
        """
        generic version of "top words" using _collect_random_sample
        """
        # support sample_size kwarg
        sample_size = kwargs.pop('sample_size', self.WORDS_SAMPLE)

        # collect titles by language
        # (spacy tokenizer has high startup cost)
        titles: dict[str, list[str]] = collections.defaultdict(list)
        title_count = 0
        for story in self._collect_random_sample(query, start_date, end_date, samples=sample_size,
                                                 fields=["title", "language"], **kwargs):
            if "title" in story:
                titles[story.get("language", 'UNK')].append(story["title"])
                title_count += 1

        # now tokenize by language, removing stopwords and tally by term & document
        term_counts: collections.Counter = collections.Counter() # total appearances of a term
        doc_counts: collections.Counter = collections.Counter()  # number of documents with a term

        for language, title_list in titles.items():
            for doc_word_list in terms_without_stopwords_list(language, title_list): # takes min_length
                this_doc_counts = collections.Counter(doc_word_list)
                term_counts.update(this_doc_counts)
                doc_counts.update(this_doc_counts.keys()) # count once per document

        # format results
        results = terms_from_counts(term_counts, doc_counts, title_count, limit)
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
            # 1. Look for OBJ_NAME_VARIABLE env var, returns value if it exists.
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

    def __incr(self, name: str) -> None:
        """
        statsd creates two files per counter.
        create a new helper (like _incr_query_op)
        for each new class of counters
        """
        self.trace(Trace.STATS, "incr %s", name)
        if self._statsd_client:
            self._statsd_client.incr(name)

    def _incr_query_op(self, op: str) -> None:
        """
        called each time a backing client is called
        op name should be related to Provider.method (with dashes)
        """
        op = op.replace("_", "-")
        self.__incr(f"query.op_{op}") # using label_value

    def _timing(self, name: str, ms: float) -> None:
        # for timings statsd makes 54 files (38MB per metric per app)
        # so easy to waste lots of space....
        raise NotImplementedError("statsd timing not yet implemented")

# used in normalized_count_over_time
def _combined_split_and_normalized_counts(matching_results: list[_DateCount], total_results: list[_DateCount]) -> list[_CombinedDateInfo]:
    counts = []
    for day in total_results:
        day_info = _CombinedDateInfo(date=day['date'], total_count=day['count'])
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
    counts.sort(key=itemgetter('date'))
    return counts

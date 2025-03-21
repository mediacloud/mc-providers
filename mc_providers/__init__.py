import logging
from typing import Any, List, NamedTuple, Optional

from .exceptions import UnknownProviderException, MissingRequiredValue
from .provider import ContentProvider, set_default_timeout
# NOTE! DEFAULT_TIMEOUT no longer imported here:
# doesn't change when set_default_timeout called!!!
from .onlinenews import OnlineNewsWaybackMachineProvider, OnlineNewsMediaCloudProvider


logger = logging.getLogger(__name__)

# static list matching topics/info results

PLATFORM_ONLINE_NEWS = 'onlinenews'

# static list matching topics/info results

PLATFORM_SOURCE_WAYBACK_MACHINE = 'waybackmachine'
PLATFORM_SOURCE_MEDIA_CLOUD = "mediacloud"     # direct to elasticsearch

NAME_SEPARATOR = "-"

def provider_name(platform: str, source: str) -> str:
    return platform + NAME_SEPARATOR + source

# map provider name to class to instantiate.
# if each class had PLATFORM_NAME and SOURCE_NAME members,
# this map could ve constructed from just a list of classes.
_PROVIDER_MAP: dict[str, type[ContentProvider]] = {
    provider_name(PLATFORM_ONLINE_NEWS, PLATFORM_SOURCE_WAYBACK_MACHINE): OnlineNewsWaybackMachineProvider,
    provider_name(PLATFORM_ONLINE_NEWS, PLATFORM_SOURCE_MEDIA_CLOUD): OnlineNewsMediaCloudProvider,
}

_PROVIDER_NAMES: List[str] = list(_PROVIDER_MAP.keys())

def available_provider_names() -> List[str]:
    # called from frontend/index.html view, so pre-calculated
    return _PROVIDER_NAMES

def provider_for(platform: str, source: str,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 timeout: Optional[int] = None,
                 caching: int = 1,
                 # PLEASE do not add any new parameters here!
                 # See provider_by_name "NOTE!" comment below
                 **kwargs: Any) -> ContentProvider:
    """
    :param platform: One of the PLATFORM_* constants above.
    :param source: One of the PLATFORM_SOURCE_* constants above.

    see provider_by_name for kwargs
    """
    return provider_by_name(provider_name(platform, source),
                            api_key, base_url, timeout, caching, **kwargs)


def provider_by_name(name: str,
                     # NOTE! the named parameters below ONLY
                     # persist to maintain type signature compatibility
                     # with existing code that passes parameters by position!
                     # PLEASE do not add any new parameters here!
                     api_key: Optional[str] = None,
                     base_url: Optional[str] = None,
                     timeout: Optional[int] = None,
                     caching: int = 1,
                     # New parameters should ONLY be added to Provider
                     # constructors.  Parameters that apply to all providers
                     # should be added to ContentProvider.__init__
                     **kwargs: Any) -> ContentProvider:
    """
    A factory method that returns the appropriate data provider. Throws an exception to let you know if the
    platform/source arguments are unsupported.

    All providers support kwargs:
    :param caching: zero to disable in-library caching
    :param timeout: override the default timeout for the provider (in seconds)

    Providers may support (among others):
    :param api_key: The API key needed to access the provider (may be required)
    :param base_url: For custom integrations you can provide an alternate base URL for the provider's API server
    :param session_id: String that identifies client session
    :param software_id: String that identifies client software

    :return: the appropriate ContentProvider subclass
    """
    platform_provider: ContentProvider

    if name not in _PROVIDER_MAP:
        platform, source = name.split(NAME_SEPARATOR, 1)
        raise UnknownProviderException(platform, source)

    return _PROVIDER_MAP[name](
        api_key=api_key, base_url=base_url,
        timeout=timeout, caching=caching,
        **kwargs
    )


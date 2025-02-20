from typing import Any

class ProviderException(Exception):
    pass


class UnsupportedOperationException(ProviderException):
    pass


class UnknownProviderException(ProviderException):
    def __init__(self, platform: str, source: str):
        super().__init__("Unknown provider {} from {}".format(platform, source))


class MissingRequiredValue(ProviderException):
    def __init__(self, name: str, keyword: str):
        super().__init__(f"provider {name} requires {keyword}")


class QueryingEverythingUnsupportedQuery(ProviderException):
    def __init__(self) -> None:
        super().__init__("Can't query everything")


# backwards compatibility:
APIKeyRequired = MissingRequiredValue
UnavailableProviderException = UnknownProviderException

class BifurcatedProviderException(ProviderException):
    """
    base class for returning both end-user friendly and detailed info
    for optional display
    """
    def __init__(self, friendly: str, detail: Any = None):
        self.friendly = friendly
        self.detail = detail

    def __str__(self) -> str:
        return self.friendly

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.friendly!r},{self.detail!r})"

class TemporaryProviderException(BifurcatedProviderException):
    """
    Query failed for a temporary reason.
    """

class PermanentProviderException(BifurcatedProviderException):
    """
    Query failed for a permanent reason.
    """

class MysteryProviderException(BifurcatedProviderException):
    """
    Query failed for a unknown reason, not known whether permanent or temporary!
    """

# subclasses of Permanent/Temporary/Mystery:
# add new ones to identify particular classes of errors, and to
# signify that an end-user-friendly string is returned by str()

class ProviderParseException(PermanentProviderException):
    """
    Query string failed to parse
    """

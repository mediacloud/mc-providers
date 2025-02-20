Media Cloud Providers Library
=============================

A package of search providers for Media Cloud, wrapping up interfaces for different social media platform.

Install with pip (`pip install .`) and the `install.sh` script. 

Requires environment variables set for various interfaces to work correctly.


### Build

Make sure `pip install flit twine` so you can build and deploy to PyPI.

1. Bump the version number in `pyproject.toml`
2. Add a note about changes to the version history below
3. Commit the changes and tag it with a semantic version number
4. A github action will build and push the repository on committing a tagged version

### Version History
* __v0.0.0__ - Testing new release pipeline, temporary, do not use. 
* __v3.1.2__ - Fix random sampling behavior in ES provider to be genuinely random, bugfix related to marginal sorting error, additional counters for fine-grained visibility 
* __v3.1.1__ - Fix ES Provider to send None as last page pagination token
* __v3.1.0__ - Add new ProviderException classes to pass more meaningful errors to consumer processes
* __v3.0.1__ - Fix ES Provider to accept sort_{order,field} paging arguments like NSA-based Provider
* __v3.0.0__ - New "OnlineNewsMediaCloudProvider" using Elasticsearch DSL for direct access to the ES cluster. Retain old provider as "OnlineNewsMediaCloudOldProvider" for now. 
* __v2.2.0__ - Added an optional argument to providers to toggle caching behavior, added more specific error on 504
* __v2.1.1__ - Bugfix
* __v2.1.0__ - Mediacloud news client code incorperated into this package
* __v2.0.5__ - Build-system in pyproject.toml
* __v2.0.4__ - reintroduce stopwords
* __v2.0.3__ - version bump for automatic releases
* __v2.0.2__ - respect domain filters on Media Cloud searches
* __v2.0.1__ - more work on caching strategies  
* __v2.0.0__ - change CachingManager interface to support online news providers better  
* __v1.0.1__ - fix default timeout option that applies across all providers 
* __v1.0.0__ - Remove legacy Media Cloud, add timeout option to `provider_for` 
* __v0.5.3__ - Temporary fix to onlinenews-mediacloud search handling 
* __v0.5.3__ - Tweaks to onlinenews-mediacloud for compatibility with new database pattern
* __v0.5.2__ - Fix to allow override of chunk'ing in MC client 
* __v0.5.1__ - Fix use of media cloud to respect domains clause on story list paging
* __v0.5.0__ - Integrate new mediacloud-news-client into onlinenews-mediacloud
* __v0.4.0__ - Specify custom base URLs via new string param to `provider_by_name` and `provider_for` 
* __v0.3.0__ - Add support for paging through stories directly, and including text in paged results for speed
* __v0.2.6__ - Fixed querying by domain on new mediacloud system
* __v0.2.5__ - Alignment with new mediacloud system. Old onlinenews provider is now "onlinenews-mclegacy", "onlinenews-mediacloud" now queries the new index.
* __v0.2.4__ - Added support for api keys via "provider_by_name"
* __v0.2.3__ - removed support for API keys in environment variables- now expected as an argument in `providers.provider_for`
* __v0.2.2__ - transition to use the dedicated mediacloud-api-legacy package to avoid version conflictsgit
* __v0.2.1__ - add in a date hack to resolve a lower-level bug in the Media Cloud legacy count-over-time results
* __v0.2.0__ - add in support for Media Cloud legacy database
* __v0.1.7__ - corrected support for a "filters" kwarg in online_news
* __v0.1.6__ - Added support for a "filters" kwarg in online_news
* __v0.1.5__ - Added politeness wait to all chunked queries in twitter provider
* __v0.1.4__ - Added Query Chunking for large collections in the Twitter provider
* __v0.1.3__ - Added Query Chunking for large queries in the onlinenews provider
* __v0.1.2__ - Test Completeness
* __v0.1.1__ - Parity with web-search module, and language model
* __v0.1.0__ - Initial pypi upload 

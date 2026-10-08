"""Public-suffix-aware SLD and suffix extraction shared by the analysis pipelines and core/tld.py."""

import tldextract

# Bundled suffix snapshot only: no network fetch, identical output on every machine.
_extractor = tldextract.TLDExtract(suffix_list_urls=())


def extract_sld(domain: str) -> str:
    """Return the registrable label left of the public suffix.

    example.co.uk -> example, mail.google.com -> google, a bare label
    passes through unchanged. Falls back to the first label when the
    input has no registrable part (e.g. it is itself a public suffix).
    """
    sld = _extractor(domain).domain
    return sld if sld else domain.split(".")[0]



def extract_suffix(domain: str) -> str:
    """Return the public suffix (TLD or multi-label suffix such as co.uk) of a domain, or "" if none."""
    return _extractor(domain).suffix

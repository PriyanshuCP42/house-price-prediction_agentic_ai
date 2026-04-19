"""
Web Search Tool — Searches the internet for live real estate data globally.
Uses DuckDuckGo (free, no API key). Region-aware with smart query building.
"""

import re
from config.settings import (
    WEB_SEARCH_CACHE_SIZE,
    WEB_SEARCH_MAX_RESULTS,
    WEB_SEARCH_TIMEOUT_SECONDS,
)

# Blocked domains — never return results from these
BLOCKED_DOMAINS = {
    "facebook.com", "twitter.com", "x.com", "instagram.com", "tiktok.com",
    "reddit.com", "pinterest.com", "youtube.com", "amazon.com", "ebay.com",
    "craigslist.org", "zhihu.com", "baidu.com", "qq.com", "weibo.com",
    "aliexpress.com", "alibaba.com", "taobao.com",
}

# Session-level search counter
_search_count = 0
MAX_SEARCHES_PER_SESSION = 15
_SEARCH_CACHE: dict[tuple[str, str, int], str] = {}

TRUSTED_REAL_ESTATE_DOMAINS = {
    "zillow.com", "redfin.com", "realtor.com", "trulia.com", "apartments.com",
    "magicbricks.com", "99acres.com", "housing.com", "makaan.com",
    "rightmove.co.uk", "zoopla.co.uk", "onthemarket.com",
    "realtor.ca", "domain.com.au", "realestate.com.au",
}

REAL_ESTATE_RESULT_TERMS = {
    "property", "home", "house", "housing", "real estate", "listing", "rent",
    "sale", "price", "sqft", "sq ft", "apartment", "flat", "condo", "villa",
    "bedroom", "bathroom", "bhk", "market", "mortgage", "neighborhood",
}

# Indian city/pincode patterns
_INDIAN_CITIES = {
    "mumbai", "delhi", "bangalore", "bengaluru", "hyderabad", "chennai",
    "pune", "kolkata", "ahmedabad", "jaipur", "lucknow", "noida", "gurgaon",
    "gurugram", "thane", "navi mumbai", "faridabad", "ghaziabad", "chandigarh",
    "kochi", "coimbatore", "indore", "bhopal", "patna", "varanasi", "agra",
    "surat", "vadodara", "nagpur", "visakhapatnam", "mysore", "mangalore",
    "trivandrum", "thiruvananthapuram", "dehradun", "ranchi", "bhubaneswar",
    "guwahati", "raipur", "ludhiana", "amritsar", "kanpur", "allahabad",
    "prayagraj", "meerut", "jodhpur", "madurai", "nashik", "rajkot",
}


def _is_blocked(url: str) -> bool:
    """Check if URL is from a blocked domain."""
    url_lower = url.lower()
    return any(d in url_lower for d in BLOCKED_DOMAINS)


def _sanitize_query(query: str) -> str:
    query = re.sub(r'[<>{}()\[\]|;`\\]', '', query)
    query = re.sub(r'https?://\S+', '', query)
    query = re.sub(r'\s+', ' ', query).strip()
    return query[:200]


def _remember_cache(key: tuple[str, str, int], value: str) -> None:
    if len(_SEARCH_CACHE) >= WEB_SEARCH_CACHE_SIZE:
        _SEARCH_CACHE.pop(next(iter(_SEARCH_CACHE)))
    _SEARCH_CACHE[key] = value


def _result_score(title: str, snippet: str, source: str, user_query: str) -> int:
    text = f"{title} {snippet}".lower()
    score = 0
    score += sum(1 for term in REAL_ESTATE_RESULT_TERMS if term in text)
    score += 4 if any(source.endswith(domain) for domain in TRUSTED_REAL_ESTATE_DOMAINS) else 0
    # Boost results that contain actual price figures (₹, $, lakh, crore, etc.)
    if re.search(r'[\$₹€£]\s*[\d,]+|\d+\s*(?:lakh|lac|crore|cr|million|k)\b|\d[\d,]*\s*/\s*(?:sqft|sq\.?\s*ft|sq\.?\s*m)', text, re.IGNORECASE):
        score += 3

    query_tokens = {
        token for token in re.findall(r"\b[a-zA-Z0-9]{3,}\b", user_query.lower())
        if token not in {"the", "and", "for", "with", "price", "property", "estate"}
    }
    score += min(4, sum(1 for token in query_tokens if token in text))
    return score


def _detect_region(query: str) -> tuple[str, str]:
    """Detect region from query. Returns (ddg_region, context_label)."""
    q = query.lower()

    # Indian signals
    if re.search(r'\b\d{6}\b', query):  # 6-digit pincode
        return "in-en", "India"
    if any(city in q for city in _INDIAN_CITIES):
        return "in-en", "India"
    if any(w in q for w in ["india", "pincode", "lakh", "crore", "bhk", "rupee"]):
        return "in-en", "India"

    # UK signals
    if any(w in q for w in ["uk", "london", "manchester", "birmingham", "edinburgh", "glasgow", "bristol"]):
        return "uk-en", "UK"

    # Canada
    if any(w in q for w in ["canada", "toronto", "vancouver", "montreal", "ottawa", "calgary"]):
        return "ca-en", "Canada"

    # Australia
    if any(w in q for w in ["australia", "sydney", "melbourne", "brisbane", "perth"]):
        return "au-en", "Australia"

    # Default: US
    return "us-en", "US"


def _build_search_query(user_query: str) -> str:
    """Build a search query with real estate context."""
    q = user_query.lower()
    region, label = _detect_region(user_query)

    # Check if query has RE keywords already
    re_terms = ["price", "home", "house", "property", "real estate", "rent",
                "flat", "apartment", "plot", "villa", "bhk", "sqft", "sq ft",
                "cost", "rate", "value", "worth", "buy", "sell", "market"]
    has_re = any(t in q for t in re_terms)

    suffix = ""
    if not has_re:
        if label == "India":
            suffix = " property price rate per sqft"
        else:
            suffix = " property price real estate"

    # For Indian pincodes, add "property" context
    pincode = re.search(r'\b(\d{6})\b', user_query)
    if pincode and "property" not in q and "price" not in q:
        suffix = " property price area"

    return f"{user_query}{suffix}"


import threading
_web_search_lock = threading.Lock()

def search_real_estate(query: str, max_results: int = 5) -> str:
    """Search the web for real estate data globally.

    Accepts ALL non-blocked results (not just whitelisted domains).
    Uses region-aware search for better results.
    """
    global _search_count

    clean_query = _sanitize_query(query)
    if len(clean_query) < 5:
        return "[Query too short for web search.]"

    max_results = max(1, min(int(max_results or 5), WEB_SEARCH_MAX_RESULTS))
    search_query = _build_search_query(clean_query)
    region, region_label = _detect_region(clean_query)
    cache_key = (search_query. lower(), region, max_results)
    if cache_key in _SEARCH_CACHE:
        return _SEARCH_CACHE[cache_key]

    with _web_search_lock:
        if _search_count >= MAX_SEARCHES_PER_SESSION:
            return "[Web search limit reached for this session.]"

        _search_count += 1

        try:
            from ddgs import DDGS

            results = []
            search_kwargs = {
                "max_results": max_results * 4,
                "region": region,
                "safesearch": "moderate",
            }
            try:
                ddgs = DDGS(timeout=WEB_SEARCH_TIMEOUT_SECONDS)
            except TypeError:
                ddgs = DDGS()
            for r in ddgs.text(search_query, **search_kwargs):
                url = r.get("href", "")

                if _is_blocked(url):
                    continue
    
                title = r.get("title", "")[:120]
                snippet = r.get("body", "")[:500]
    
                # Skip results with non-Latin scripts (Chinese, Arabic, etc.)
                if re.search(r'[\u4e00-\u9fff\u0600-\u06ff]', snippet):
                    continue
    
                domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
                source = domain_match.group(1) if domain_match else "web"
                score = _result_score(title, snippet, source, clean_query)
                if score < 1:
                    continue
                results.append((score, f"[{source}] {title}\n{snippet}"))
    
                if len(results) >= max_results:
                    break
    
            if not results:
                return (
                    f"[No real estate results found for this query. "
                    f"Try being more specific — include the city name, property type (house/flat/apartment), "
                    f"and number of bedrooms.]"
                )
    
            ranked = [text for _, text in sorted(results, key=lambda item: item[0], reverse=True)]
            response = f"Live Web Search Results ({len(ranked)} results, region: {region_label}):\n\n" + "\n\n".join(ranked)
            _remember_cache(cache_key, response)
            return response
    
        except ImportError:
            return "[Web search unavailable — install duckduckgo-search package.]"
        except TypeError:
            return "[Web search temporarily unavailable due to provider option mismatch. Try again.]"
        except Exception as error:
            return f"[Web search temporarily unavailable. Try again. Details: {type(error).__name__}]"


def get_search_count() -> int:
    return _search_count


def reset_search_count():
    global _search_count
    _search_count = 0


def clear_search_cache():
    _SEARCH_CACHE.clear()

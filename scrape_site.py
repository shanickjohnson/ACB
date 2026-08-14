"""
Scrapes ACB Caribbean's public website into web_content.json, in a format
rag.py can chunk and embed alongside the existing fee/service JSON files.

Usage:
    pip install requests beautifulsoup4 --break-system-packages
    python scrape_site.py

This is meant to be run occasionally (e.g. whenever the marketing site
changes), not on every server startup -- it commits its output
(web_content.json) to the repo like the other knowledge files, and rag.py
just reads that file like any other source.
"""

import json
import time
import urllib.robotparser
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

USER_AGENT = "ACB-KnowledgeBase-Bot/1.0 (internal chatbot content sync)"
REQUEST_DELAY_SECONDS = 1.5  # be polite -- don't hammer the site

# Curated list of public, informational pages worth adding to the knowledge
# base. Deliberately excludes: secure.acbonline.com (login-gated), PDFs
# (need separate extraction -- see note at the bottom), and pages that are
# pure navigation/marketing fluff with no factual content to ground answers.
PAGES = {
    "Antigua & Barbuda": [
        "https://ag.acbonline.com/personal/banking/",
        "https://ag.acbonline.com/personal/banking/acb-digital-banking/",
        "https://ag.acbonline.com/personal/transfers/",
        "https://ag.acbonline.com/personal/mortgage/",
        "https://ag.acbonline.com/personal/student-loans/",
        "https://ag.acbonline.com/personal/vehicle-loans/",
        "https://ag.acbonline.com/personal/land-loans/",
        "https://ag.acbonline.com/personal/unsecured-loans/",
        "https://ag.acbonline.com/personal/personal-loans-travel-vacation/",
        "https://ag.acbonline.com/personal/home-equity-loans-renovations/",
        "https://ag.acbonline.com/personal/personal-chequing-account/",
        "https://ag.acbonline.com/personal/acb-smart-access-account/",
        "https://ag.acbonline.com/personal/acb-smart-cards/",
        "https://ag.acbonline.com/personal/freedom-55/",
        "https://ag.acbonline.com/personal/junior-savings/",
        "https://ag.acbonline.com/personal/savings-account-2/",
        "https://ag.acbonline.com/personal/acb-caribbean-credit-cards/",
        "https://ag.acbonline.com/business/general-banking/",
        "https://ag.acbonline.com/business/business-chequing/",
        "https://ag.acbonline.com/business/corporate-credit-cards/",
        "https://ag.acbonline.com/business/merchant-services/",
        "https://ag.acbonline.com/business/business-loans/",
        "https://ag.acbonline.com/mortgage-and-trust/",
        "https://ag.acbonline.com/mortgage-and-trust/fixed-deposits/",
        "https://ag.acbonline.com/mortgage-and-trust/thrift-fund/",
        "https://ag.acbonline.com/acb-caribbean/",
        "https://ag.acbonline.com/acb-caribbean/contact-us/",
        "https://ag.acbonline.com/acb-caribbean/location/",
        "https://ag.acbonline.com/faqs/",
        "https://ag.acbonline.com/glossary/",
        "https://ag.acbonline.com/careers/",
    ],
    "Grenada": [
        "https://gd.acbonline.com/",
        # Add Grenada-specific equivalents of the above as they're confirmed --
        # the subdomain's page structure may not mirror Antigua's exactly.
    ],
}


def robots_allows(url: str) -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
    except Exception:
        # If robots.txt can't be fetched, err on the side of not scraping.
        print(f"Warning: could not read {robots_url}, skipping {url}")
        return False
    return rp.can_fetch(USER_AGENT, url)


def extract_main_content(html: str) -> tuple[str, str]:
    """Returns (title, body_text), stripped of nav/header/footer/script noise."""
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    # WordPress-specific chrome that isn't page content
    for selector in [".menu", ".navbar", ".site-footer", ".cookie", "#cookie-notice"]:
        for el in soup.select(selector):
            el.decompose()

    main = soup.find("main") or soup.find(attrs={"role": "main"}) or soup.body or soup
    text = main.get_text(separator=" ", strip=True)
    # Collapse repeated whitespace left over from stripped tags
    text = " ".join(text.split())
    return title, text


def scrape() -> dict:
    pages_out = []
    for jurisdiction, urls in PAGES.items():
        for url in urls:
            if not robots_allows(url):
                print(f"Skipping (robots.txt disallows): {url}")
                continue
            try:
                resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
                resp.raise_for_status()
            except Exception as e:
                print(f"Failed to fetch {url}: {e}")
                continue

            title, text = extract_main_content(resp.text)
            if len(text) < 100:
                print(f"Skipping (too little content, likely a JS-rendered page): {url}")
                continue

            pages_out.append({
                "url": url,
                "jurisdiction": jurisdiction,
                "title": title,
                "text": text,
            })
            print(f"Scraped: {url} ({len(text)} chars)")
            time.sleep(REQUEST_DELAY_SECONDS)

    return {"pages": pages_out}


if __name__ == "__main__":
    result = scrape()
    with open("web_content.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(result['pages'])} pages to web_content.json")

# ---------------------------------------------------------------------------
# Note on PDFs (fee schedule, digital banking guide, etc.):
# The site links to PDFs like the "ACB Caribbean Digital Banking Guide" and
# possibly a fee schedule PDF. Those need a different extraction path (e.g.
# pypdf or pdfplumber) since BeautifulSoup can't read PDF content. If you
# want those included too, that's a small follow-up script -- worth doing
# separately since PDF text extraction quality varies and should be
# spot-checked before feeding it to the RAG index.
# ---------------------------------------------------------------------------
